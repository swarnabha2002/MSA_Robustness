"""
Enhanced MISA models.py
Drop-in replacement for MSA-Robustness/MISA/src/models.py

Changes vs original:
  [A] TransformerEncoder replaces GRU for visual + acoustic streams
  [B] Modality dropout augmentation (training-time)
  [C] Cross-modal attention across the 3 invariant (shared) representations
  [D] Uncertainty-aware gating — learned scalar confidence per modality
  [E] NT-Xent contrastive alignment loss exposed via get_contrastive_loss()
  
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from utils import to_gpu, ReverseLayerF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def masked_mean(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-9)


def masked_max(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


# ---------------------------------------------------------------------------
# [A] Lightweight Transformer encoder for a single modality
#     Replaces the two-layer bidirectional GRU used for audio & video
# ---------------------------------------------------------------------------

class ModalityTransformerEncoder(nn.Module):
    """
    Projects raw modality features to hidden_size, then runs
    a 2-layer Transformer encoder, then mean-pools over the sequence.

    Input  : (seq_len, batch, input_size)   — same convention as GRU
    Output : (batch, hidden_size)
    """
    def __init__(self, input_size, hidden_size, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        # project raw features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=False,   # (seq, batch, d)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, src_key_padding_mask=None):
        # x: (seq_len, batch, input_size)
        x = self.input_proj(x)                                  # (seq, batch, H)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (seq, batch, H)
        x = x.mean(dim=0)                                       # (batch, H)
        return self.norm(x)


# ---------------------------------------------------------------------------
# [C] Cross-modal attention across three invariant representations
#     Each modality's shared repr attends to the other two as context
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    Given three vectors (h_t, h_v, h_a) — each (batch, hidden_size) —
    produce three refined vectors where each one has attended to the others.

    Operates in sequence dimension = 1 (the three modalities form a 3-token sequence),
    so MultiheadAttention sees (seq=3, batch, hidden_size).
    """
    def __init__(self, hidden_size, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h_t, h_v, h_a):
        # stack: (3, batch, H)
        x = torch.stack([h_t, h_v, h_a], dim=0)
        out, _ = self.attn(x, x, x)           # self-attention across modalities
        out = self.norm(x + out)               # residual + norm
        return out[0], out[1], out[2]          # (batch, H) each


# ---------------------------------------------------------------------------
# [D] Uncertainty-aware gating
#     Each modality produces a scalar confidence; softmax weights fusion
# ---------------------------------------------------------------------------

class UncertaintyGate(nn.Module):
    """
    Learns a scalar confidence score from each modality's combined representation.
    Returns a soft (differentiable) weight for each modality.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # one small MLP per modality → scalar logit
        self.gate_t = nn.Linear(hidden_size, 1)
        self.gate_v = nn.Linear(hidden_size, 1)
        self.gate_a = nn.Linear(hidden_size, 1)

    def forward(self, h_t, h_v, h_a):
        logits = torch.cat([self.gate_t(h_t),
                             self.gate_v(h_v),
                             self.gate_a(h_a)], dim=-1)   # (batch, 3)
        weights = torch.softmax(logits, dim=-1)            # (batch, 3)
        return weights                                     # w_t, w_v, w_a per sample


# ---------------------------------------------------------------------------
# Main enhanced MISA model
# ---------------------------------------------------------------------------

class MISA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.text_size     = config.embedding_size
        self.visual_size   = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes  = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size  = config.num_classes
        self.dropout_rate = config.dropout

        self.activation = config.activation()
        self.tanh = nn.Tanh()

        # ------------------------------------------------------------------ #
        # Text encoder (unchanged — BERT or GRU as in original)
        # ------------------------------------------------------------------ #
        if config.use_bert:
            bertconfig = BertConfig.from_pretrained(
                'bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained(
                'bert-base-uncased', config=bertconfig)
        else:
            rnn = nn.LSTM if config.rnncell == "lstm" else nn.GRU
            self.embed  = nn.Embedding(len(config.word2id), self.text_size)
            self.trnn1  = rnn(self.text_size, self.hidden_sizes[0], bidirectional=True)
            self.trnn2  = rnn(2*self.hidden_sizes[0], self.hidden_sizes[0], bidirectional=True)
            self.tlayer_norm = nn.LayerNorm((self.hidden_sizes[0]*2,))

        # ------------------------------------------------------------------ #
        # [A] Transformer encoders for visual & acoustic (replaces GRU)
        # ------------------------------------------------------------------ #
        self.visual_transformer = ModalityTransformerEncoder(
            input_size  = self.visual_size,
            hidden_size = config.hidden_size,
            nhead       = 4,
            num_layers  = 2,
            dropout     = self.dropout_rate,
        )
        self.acoustic_transformer = ModalityTransformerEncoder(
            input_size  = self.acoustic_size,
            hidden_size = config.hidden_size,
            nhead       = 4,
            num_layers  = 2,
            dropout     = self.dropout_rate,
        )

        # ------------------------------------------------------------------ #
        # Text projection (maps BERT 768 or GRU hidden → hidden_size)
        # Visual / acoustic projection is now inside ModalityTransformerEncoder
        # ------------------------------------------------------------------ #
        if config.use_bert:
            self.project_t = nn.Sequential(
                nn.Linear(768, config.hidden_size),
                config.activation(),
                nn.LayerNorm(config.hidden_size),
            )
        else:
            self.project_t = nn.Sequential(
                nn.Linear(self.hidden_sizes[0]*4, config.hidden_size),
                config.activation(),
                nn.LayerNorm(config.hidden_size),
            )
        # v and a projections are now integrated in the Transformer encoders above

        # ------------------------------------------------------------------ #
        # Private (modality-specific) encoders
        # ------------------------------------------------------------------ #
        self.private_t = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )
        self.private_v = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )
        self.private_a = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------ #
        # Shared (modality-invariant) encoder
        # ------------------------------------------------------------------ #
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------ #
        # Reconstruction heads
        # ------------------------------------------------------------------ #
        self.recon_t = nn.Linear(config.hidden_size, config.hidden_size)
        self.recon_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.recon_a = nn.Linear(config.hidden_size, config.hidden_size)

        # ------------------------------------------------------------------ #
        # Shared-space adversarial discriminator (unchanged)
        # ------------------------------------------------------------------ #
        if not config.use_cmd_sim:
            self.discriminator = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(config.hidden_size, len(self.hidden_sizes)),
            )

        self.sp_discriminator = nn.Sequential(
            nn.Linear(config.hidden_size, 4),
        )

        # ------------------------------------------------------------------ #
        # [C] Cross-modal attention on invariant representations
        # ------------------------------------------------------------------ #
        self.cross_modal_attn = CrossModalAttention(
            hidden_size = config.hidden_size,
            nhead       = 4,
            dropout     = self.dropout_rate,
        )

        # ------------------------------------------------------------------ #
        # [D] Uncertainty gating
        # ------------------------------------------------------------------ #
        self.uncertainty_gate = UncertaintyGate(config.hidden_size)

        # ------------------------------------------------------------------ #
        # Fusion head
        # Input: concat of [private_t, private_v, private_a,
        #                    refined_shared_t, refined_shared_v, refined_shared_a]
        #        = hidden_size * 6    (same shape as original — compatible)
        # ------------------------------------------------------------------ #
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 6, config.hidden_size * 3),
            nn.Dropout(self.dropout_rate),
            self.activation,
            nn.Linear(config.hidden_size * 3, self.output_size),
        )

        # Kept for compatibility with original code paths
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=2),
            num_layers=1,
        )

        # storage for loss computation (populated in forward)
        self.utt_private_t = None
        self.utt_private_v = None
        self.utt_private_a = None
        self.utt_shared_t  = None
        self.utt_shared_v  = None
        self.utt_shared_a  = None

    # ---------------------------------------------------------------------- #
    # GRU feature extractor (used only when use_bert=False, text modality)
    # ---------------------------------------------------------------------- #
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed = pack_padded_sequence(sequence, lengths)
        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed)
        else:
            packed_h1, final_h1 = rnn1(packed)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed = pack_padded_sequence(normed_h1, lengths)
        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed)
        else:
            _, final_h2 = rnn2(packed_normed)
        return final_h1, final_h2

    # ---------------------------------------------------------------------- #
    # [B] Modality dropout — zeroes an entire modality with prob p
    # ---------------------------------------------------------------------- #
    def modality_dropout(self, h_t, h_v, h_a, p=0.3):
        """
        During training, randomly zero out one complete modality per sample.
        p = probability of dropping any given modality (independent per modality).
        We never drop all three simultaneously.
        """
        if not self.training:
            return h_t, h_v, h_a

        batch = h_t.size(0)
        # sample masks: 1 = keep, 0 = drop
        mask_t = (torch.rand(batch, 1, device=h_t.device) > p).float()
        mask_v = (torch.rand(batch, 1, device=h_v.device) > p).float()
        mask_a = (torch.rand(batch, 1, device=h_a.device) > p).float()

        # safety: if all three are dropped for a sample, keep text
        all_dropped = (mask_t * mask_v * mask_a == 0).float().unsqueeze(1)  # but this checks products
        # simpler: ensure at least one is always 1
        # find samples where all are 0
        all_zero = ((mask_t + mask_v + mask_a) == 0).float()
        mask_t = mask_t + all_zero   # restore text if everything is 0
        mask_t = mask_t.clamp(0, 1)

        return h_t * mask_t, h_v * mask_v, h_a * mask_a

    # ---------------------------------------------------------------------- #
    # Alignment (main forward computation)
    # ---------------------------------------------------------------------- #
    def alignment(self, sentences, visual, acoustic, lengths,
                  bert_sent, bert_sent_type, bert_sent_mask):

        batch_size = lengths.size(0)

        # ---- Text encoding -------------------------------------------- #
        if self.config.use_bert:
            bert_out = self.bertmodel(
                input_ids      = bert_sent,
                attention_mask = bert_sent_mask,
                token_type_ids = bert_sent_type,
            )[0]
            masked_out = torch.mul(bert_sent_mask.unsqueeze(2), bert_out)
            mask_len   = torch.sum(bert_sent_mask, dim=1, keepdim=True).float()
            utterance_text = torch.sum(masked_out, dim=1) / mask_len.clamp(min=1e-9)
            utterance_text = self.project_t(utterance_text)           # (B, H)
        else:
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(
                sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2) \
                               .permute(1, 0, 2).contiguous().view(batch_size, -1)
            utterance_text = self.project_t(utterance_text)

        # ---- [A] Visual encoding (Transformer) ------------------------- #
        # visual: (seq_len, batch, visual_size)  — standard layout from dataloader
        # Build padding mask: True = position is padding (to be ignored)
        lengths_cpu = lengths.cpu()
        visual_key_padding_mask = torch.zeros(
            batch_size, visual.size(0), dtype=torch.bool, device=visual.device)
        for i, l in enumerate(lengths_cpu):
            if l < visual.size(0):
                visual_key_padding_mask[i, l:] = True
        # ModalityTransformerEncoder expects (seq, batch, feat)
        utterance_video = self.visual_transformer(
            visual, src_key_padding_mask=visual_key_padding_mask)     # (B, H)

        # ---- [A] Acoustic encoding (Transformer) ----------------------- #
        acoustic_key_padding_mask = torch.zeros(
            batch_size, acoustic.size(0), dtype=torch.bool, device=acoustic.device)
        for i, l in enumerate(lengths_cpu):
            if l < acoustic.size(0):
                acoustic_key_padding_mask[i, l:] = True
        utterance_audio = self.acoustic_transformer(
            acoustic, src_key_padding_mask=acoustic_key_padding_mask) # (B, H)

        # ---- [B] Modality dropout ------------------------------------- #
        utterance_text, utterance_video, utterance_audio = self.modality_dropout(
            utterance_text, utterance_video, utterance_audio, p=0.25)

        # ---- Disentanglement: private + shared ----------------------- #
        self.utt_private_t = self.private_t(utterance_text)
        self.utt_private_v = self.private_v(utterance_video)
        self.utt_private_a = self.private_a(utterance_audio)

        self.utt_shared_t  = self.shared(utterance_text)
        self.utt_shared_v  = self.shared(utterance_video)
        self.utt_shared_a  = self.shared(utterance_audio)

        # ---- [C] Cross-modal attention on invariant representations --- #
        refined_shared_t, refined_shared_v, refined_shared_a = self.cross_modal_attn(
            self.utt_shared_t, self.utt_shared_v, self.utt_shared_a)

        # ---- [D] Uncertainty-aware gating ----------------------------- #
        # Gate uses the private repr of each modality as confidence signal
        weights = self.uncertainty_gate(
            self.utt_private_t, self.utt_private_v, self.utt_private_a)  # (B, 3)
        w_t = weights[:, 0:1]   # (B, 1)
        w_v = weights[:, 1:2]
        w_a = weights[:, 2:3]

        gated_shared_t = refined_shared_t * w_t
        gated_shared_v = refined_shared_v * w_v
        gated_shared_a = refined_shared_a * w_a

        # ---- Fusion --------------------------------------------------- #
        h = torch.cat([
            self.utt_private_t,
            self.utt_private_v,
            self.utt_private_a,
            gated_shared_t,
            gated_shared_v,
            gated_shared_a,
        ], dim=1)                                    # (B, H*6)

        o = self.fusion(h)
        return o

    def forward(self, sentences, video, acoustic, lengths,
                bert_sent, bert_sent_type, bert_sent_mask):
        o = self.alignment(sentences, video, acoustic, lengths,
                           bert_sent, bert_sent_type, bert_sent_mask)
        return o

    # ---------------------------------------------------------------------- #
    # Loss helpers (called from solver.py — same interface as original)
    # ---------------------------------------------------------------------- #

    def get_recon_loss(self):
        loss = F.mse_loss(self.recon_t(self.utt_shared_t), self.utt_private_t.detach()) + \
               F.mse_loss(self.recon_v(self.utt_shared_v), self.utt_private_v.detach()) + \
               F.mse_loss(self.recon_a(self.utt_shared_a), self.utt_private_a.detach())
        return loss / 3.0

    def get_diff_loss(self):
        """Orthogonality loss — private and shared spaces should be orthogonal."""
        def diff(a, b):
            a = a - a.mean(dim=0, keepdim=True)
            b = b - b.mean(dim=0, keepdim=True)
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            corr = (a * b).sum(dim=1).pow(2).mean()
            return corr
        return (diff(self.utt_private_t, self.utt_shared_t) +
                diff(self.utt_private_v, self.utt_shared_v) +
                diff(self.utt_private_a, self.utt_shared_a)) / 3.0

    # [E] NT-Xent contrastive alignment loss
    def get_contrastive_loss(self, temperature=0.5):
        """
        Pulls invariant representations of the same sample together across modalities.
        Applied to all three modality pairs: (T,V), (T,A), (V,A).
        """
        def nt_xent(z1, z2):
            B = z1.size(0)
            z = torch.cat([z1, z2], dim=0)              # (2B, H)
            z = F.normalize(z, dim=1)
            sim = torch.mm(z, z.T) / temperature         # (2B, 2B)
            mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
            sim = sim.masked_fill(mask, -1e9)
            labels = torch.cat([
                torch.arange(B, 2*B, device=z.device),
                torch.arange(0,  B,  device=z.device),
            ])
            return F.cross_entropy(sim, labels)

        loss  = nt_xent(self.utt_shared_t, self.utt_shared_v)
        loss += nt_xent(self.utt_shared_t, self.utt_shared_a)
        loss += nt_xent(self.utt_shared_v, self.utt_shared_a)
        return loss / 3.0

    def get_domain_loss(self):
        if self.config.use_cmd_sim:
            return torch.zeros(1, device=next(self.parameters()).device)
        # Original discriminator-based domain loss
        shared_t = ReverseLayerF.apply(self.utt_shared_t, 0.1)
        shared_v = ReverseLayerF.apply(self.utt_shared_v, 0.1)
        shared_a = ReverseLayerF.apply(self.utt_shared_a, 0.1)

        t_logits = self.discriminator(shared_t)
        v_logits = self.discriminator(shared_v)
        a_logits = self.discriminator(shared_a)

        t_labels = torch.zeros(t_logits.size(0), dtype=torch.long, device=t_logits.device)
        v_labels = torch.ones (v_logits.size(0), dtype=torch.long, device=v_logits.device)
        a_labels = 2 * torch.ones(a_logits.size(0), dtype=torch.long, device=a_logits.device)

        return (F.cross_entropy(t_logits, t_labels) +
                F.cross_entropy(v_logits, v_labels) +
                F.cross_entropy(a_logits, a_labels)) / 3.0

    def get_cmd_loss(self):
        """CMD moment-matching loss (same as original)."""
        if not self.config.use_cmd_sim:
            return torch.zeros(1, device=next(self.parameters()).device)
        def cmd(x1, x2, n_moments=5):
            loss = torch.mean(x1, dim=0) - torch.mean(x2, dim=0)
            loss = loss.norm()
            for i in range(1, n_moments):
                c1 = (x1 ** (i+1)).mean(dim=0)
                c2 = (x2 ** (i+1)).mean(dim=0)
                loss += (c1 - c2).norm()
            return loss
        return (cmd(self.utt_shared_t, self.utt_shared_v) +
                cmd(self.utt_shared_t, self.utt_shared_a) +
                cmd(self.utt_shared_v, self.utt_shared_a)) / 3.0
