"""
models/model.py — PEFT-UMamba (implementation matching the manuscript)

Paper section → code
────────────────────
  §3.2  VMamba-Tiny encoder, depths {2,2,9,2}, dims {96,192,384,768}, K=16,
        ImageNet-1K pretrained, ALL weights frozen          → VMambaEncoder
        (parameter names follow the original VMamba release, so the official
        checkpoint loads by name; loading is verified and fails loudly)
  §3.3–3.4  Supplementary Scan: K → K+K′ inside every SS2D, block-diagonal
        Â=[A_frz 0; 0 A_supp], B̂=[B_frz; B_supp], Ĉ=[C_frz C_supp],
        shared input projection and shared Δ (Eq. 5–10)     → SS2D
        Implemented by concatenating along the state axis and running ONE
        selective-scan kernel, which is exactly the block-diagonal system.
  §3.5  Neighbourhood Initialisation: A_supp^(j) = A_frz^(K) + ε (Eq. 11),
        B_supp Xavier, C_supp = 0 → network equals the frozen backbone at
        initialisation (Eq. 13)                             → SS2D.reset_supplementary
  §3.6  Symmetric VSS-block decoder with Supplementary Scan  → DecoderLayer
  §3.6.1 SCAM: bidirectional GRU over ordered slices/frames,
        f̃_t = →h_t + ←h_t (Eq. 12)                          → SCAM
  §1/§4.2 Skip Attention Gates following Oktay et al. [31]  → SkipAttentionGate
  §4.2  Deep supervision at three decoder resolutions        → aux heads
  Table 6 baselines: frozen encoder (K′=0), LoRA, full fine-tuning;
        Mona / LoRand via peft_baselines.py (blocks are named VSSBlock /
        self_attention, which that module expects).
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:                                    # pragma: no cover
    selective_scan_fn = None
    HAS_MAMBA_CUDA = False


# =========================================================================== #
#  Selective scan (CUDA kernel, with a slow-but-correct reference fallback)
# =========================================================================== #

def _selective_scan_reference(u, delta, A, B, C, D=None, delta_bias=None,
                              delta_softplus=True):
    """
    Exact sequential reference of Mamba's selective scan.
      u, delta : [b, d, l]      A : [d, n]
      B, C     : [b, g, n, l]   (g groups; d must be divisible by g)
      D        : [d]
    Returns y : [b, d, l].  Slow (Python loop over l) — only for CPU tests.
    """
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, l = u.shape
    g = B.shape[1]
    B = B.repeat_interleave(d // g, dim=1)            # [b, d, n, l]
    C = C.repeat_interleave(d // g, dim=1)
    dA = torch.exp(delta.unsqueeze(2) * A[None, :, :, None])        # [b,d,n,l]
    dBu = delta.unsqueeze(2) * B * u.unsqueeze(2)                   # [b,d,n,l]
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for t in range(l):
        h = dA[..., t] * h + dBu[..., t]
        ys.append((h * C[..., t]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None] * u
    return y


def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus=True):
    if HAS_MAMBA_CUDA and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D, z=None,
                                 delta_bias=delta_bias,
                                 delta_softplus=delta_softplus,
                                 return_last_state=False)
    return _selective_scan_reference(u, delta, A, B, C, D, delta_bias,
                                     delta_softplus)


# =========================================================================== #
#  Basic layers (names follow the original VMamba release)
# =========================================================================== #

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed2D(nn.Module):
    """[B,3,H,W] → [B,H/4,W/4,C]"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class PatchMerging2D(nn.Module):
    """[B,H,W,C] → [B,H/2,W/2,2C]"""
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x))


class PatchExpand2D(nn.Module):
    """[B,H,W,C] → [B,2H,2W,C/2]"""
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x).view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * H, 2 * W, C // 2)
        return self.norm(x)


class FinalPatchExpand2D(nn.Module):
    """[B,H/4,W/4,C] → [B,H,W,C]"""
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x).view(B, H, W, 4, 4, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 4 * H, 4 * W, C)
        return self.norm(x)


# =========================================================================== #
#  SS2D with Supplementary Scan  (§3.3–3.5)
# =========================================================================== #

class SS2D(nn.Module):
    """
    VMamba SS2D (4 scan directions, per-direction frozen x/Δ projections and
    A, D), extended with K′ supplementary state dimensions.

    Supplementary parameters (trainable):
      x_proj_supp  [2K′, d_inner]  → B_supp (first K′ rows), C_supp (last K′)
      A_logs_supp  [d_inner, K′]
    They are shared across the four scan directions (0.17M in total for the
    network at K′=4), use the frozen Δ_t of each direction (Eq. 10) and the
    same input x_t (shared input projection, §3.7).
    """

    K_DIRS = 4

    def __init__(self, d_model, d_state=16, d_state_supp=0, d_conv=3,
                 expand=2.0, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 bias=False, conv_bias=True, ni_noise_std=1e-6):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_state_supp = d_state_supp
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.ni_noise_std = ni_noise_std
        K, D, N, R = self.K_DIRS, self.d_inner, d_state, self.dt_rank

        # ── frozen (pretrained) parameters — VMamba naming ────────────────────
        self.in_proj = nn.Linear(d_model, 2 * D, bias=bias)
        self.conv2d = nn.Conv2d(D, D, d_conv, padding=(d_conv - 1) // 2,
                                groups=D, bias=conv_bias)
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.empty(K, R + 2 * N, D))
        nn.init.uniform_(self.x_proj_weight, -D ** -0.5, D ** -0.5)

        dt_std = R ** -0.5
        self.dt_projs_weight = nn.Parameter(
            torch.empty(K, D, R).uniform_(-dt_std, dt_std))
        dt = torch.exp(torch.rand(K, D) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        self.dt_projs_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(K * D, 1)
        self.A_logs = nn.Parameter(torch.log(A))           # [K·D, N]
        self.Ds = nn.Parameter(torch.ones(K * D))          # [K·D]
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, d_model, bias=bias)

        # ── supplementary (trainable) parameters ──────────────────────────────
        if d_state_supp > 0:
            self.x_proj_supp = nn.Parameter(torch.zeros(2 * d_state_supp, D))
            self.A_logs_supp = nn.Parameter(torch.zeros(D, d_state_supp))
            self.reset_supplementary("neighbourhood")

    # ── initialisation of the supplementary state (§3.5) ──────────────────────
    @torch.no_grad()
    def reset_supplementary(self, strategy: str = "neighbourhood"):
        """
        neighbourhood : A_supp^(j) = A_frz^(K) + ε^(j), ε ~ N(0, σ²)  (Eq. 11)
                        A_frz^(K) = learned value of the last frozen state
                        dimension (averaged over the 4 scan directions).
                        Must be called AFTER the pretrained weights are loaded.
        s4d           : A_supp^(j) = −(K + j)   (standard S4D continuation;
                        used for the "w/o Neighbourhood Init" ablation)
        B_supp : Xavier uniform.   C_supp : 0  → zero contribution at init.
        """
        if self.d_state_supp == 0:
            return
        Kp, D, N = self.d_state_supp, self.d_inner, self.d_state
        if strategy == "neighbourhood":
            A_frz = -torch.exp(self.A_logs.float()).view(self.K_DIRS, D, N)
            anchor = A_frz[:, :, N - 1].mean(0)                       # [D]
            A_sup = anchor[:, None] + torch.randn(D, Kp) * self.ni_noise_std
            A_sup = A_sup.clamp(max=-1e-4)                            # keep A<0
            self.A_logs_supp.copy_(torch.log(-A_sup))
        elif strategy == "s4d":
            A_sup = torch.arange(N + 1, N + Kp + 1, dtype=torch.float32)
            self.A_logs_supp.copy_(torch.log(A_sup).repeat(D, 1))
        else:
            raise ValueError(f"unknown supplementary init: {strategy}")
        self.x_proj_supp.zero_()
        nn.init.xavier_uniform_(self.x_proj_supp[:Kp])                # B_supp
        # rows Kp: (C_supp) stay exactly zero

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [B,H,W,C]
        B, H, W, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = self.act(self.conv2d(x_in.permute(0, 3, 1, 2).contiguous()))
        y = self._forward_core(x_in)                              # [B,H,W,D]
        y = self.out_norm(y)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:    # [B,D,H,W]
        B, D, H, W = x.shape
        L, K, N, Kp = H * W, self.K_DIRS, self.d_state, self.d_state_supp

        x_hwwh = torch.stack([x.view(B, -1, L),
                              x.transpose(2, 3).contiguous().view(B, -1, L)],
                             dim=1)                                # [B,2,D,L]
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B,4,D,L]

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, N, N], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)
        As = -torch.exp(self.A_logs.float())                      # [K·D, N]

        if Kp > 0:   # block-diagonal expansion K → K+K′ (Eq. 5)
            sup = torch.einsum("bkdl,cd->bkcl", xs, self.x_proj_supp)
            B_sup, C_sup = sup.split([Kp, Kp], dim=2)
            Bs = torch.cat([Bs, B_sup], dim=2)
            Cs = torch.cat([Cs, C_sup], dim=2)
            A_sup = -torch.exp(self.A_logs_supp.float()).repeat(K, 1)  # [K·D,K′]
            As = torch.cat([As, A_sup], dim=1)                     # [K·D,N+K′]

        out = selective_scan(
            xs.float().reshape(B, -1, L),
            dts.float().reshape(B, -1, L),
            As.contiguous(),
            Bs.float().contiguous(), Cs.float().contiguous(),
            self.Ds.float(),
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
        ).view(B, K, -1, L)

        inv = torch.flip(out[:, 2:4], dims=[-1])
        wh = out[:, 1].view(B, -1, W, H).transpose(2, 3).reshape(B, -1, L)
        invwh = inv[:, 1].view(B, -1, W, H).transpose(2, 3).reshape(B, -1, L)
        y = out[:, 0] + inv[:, 0] + wh + invwh                     # [B,D,L]
        return y.transpose(1, 2).reshape(B, H, W, -1).to(x.dtype)


class VSSBlock(nn.Module):
    """LN → SS2D → residual (VMamba-Tiny v0 block, no MLP)."""
    def __init__(self, dim, d_state=16, d_state_supp=0, drop_path=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attention = SS2D(dim, d_state=d_state,
                                   d_state_supp=d_state_supp)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


# =========================================================================== #
#  Encoder (§3.2)
# =========================================================================== #

class EncoderLayer(nn.Module):
    def __init__(self, dim, depth, d_state, d_state_supp, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList(
            VSSBlock(dim, d_state, d_state_supp) for _ in range(depth))
        self.downsample = PatchMerging2D(dim) if downsample else None


class VMambaEncoder(nn.Module):
    """Returns the four stage features (before down-sampling) for skips."""
    def __init__(self, depths=(2, 2, 9, 2), dims=(96, 192, 384, 768),
                 d_state=16, d_state_supp=4):
        super().__init__()
        self.patch_embed = PatchEmbed2D(4, 3, dims[0])
        self.layers = nn.ModuleList(
            EncoderLayer(dims[i], depths[i], d_state, d_state_supp,
                         downsample=(i < len(depths) - 1))
            for i in range(len(depths)))

    def forward(self, x) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        feats = []
        for layer in self.layers:
            for blk in layer.blocks:
                x = blk(x)
            feats.append(x)
            if layer.downsample is not None:
                x = layer.downsample(x)
        return feats            # [H/4,96], [H/8,192], [H/16,384], [H/32,768]


# =========================================================================== #
#  SCAM (§3.6.1), Skip Attention Gate (Oktay et al.), decoder (§3.6)
# =========================================================================== #

class SCAM(nn.Module):
    """
    Sequential Context Aggregation Module.
    Bottleneck features of an ordered sequence (adjacent slices / frames)
    are pooled, projected to a small hidden size and passed through a
    bidirectional GRU; f̃_t = →h_t + ←h_t (Eq. 12) is projected back and
    added to every spatial position.  The output projection is zero-initialised
    so SCAM starts as an identity mapping.

    ordered=True : the whole batch is ONE ordered sequence (the loader must
                   provide consecutive slices/frames in order).
    ordered=False: every sample is a length-1 sequence (unordered images).
    """
    def __init__(self, dim=768, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, f: torch.Tensor, ordered: bool) -> torch.Tensor:
        B = f.shape[0]
        v = self.down(self.norm(f.mean(dim=(1, 2))))               # [B,h]
        seq = v.unsqueeze(0) if ordered else v.unsqueeze(1)         # [1,B,h] | [B,1,h]
        out, _ = self.gru(seq.float())
        fwd, bwd = out.chunk(2, dim=-1)
        ctx = (fwd + bwd).reshape(B, -1).to(f.dtype)               # Eq. 12
        return f + self.up(ctx)[:, None, None, :]


class SkipAttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018):
       α = σ(ψ(ReLU(W_x x + W_g g))),  output = α ⊙ x"""
    def __init__(self, skip_dim, gate_dim, inter_dim=None):
        super().__init__()
        inter = inter_dim or max(skip_dim // 2, 16)
        self.W_x = nn.Linear(skip_dim, inter, bias=False)
        self.W_g = nn.Linear(gate_dim, inter, bias=True)
        self.psi = nn.Linear(inter, 1, bias=True)

    def forward(self, x, g):                                      # channel-last
        alpha = torch.sigmoid(self.psi(F.relu(self.W_x(x) + self.W_g(g))))
        return x * alpha


class DecoderLayer(nn.Module):
    def __init__(self, in_dim, skip_dim, depth, d_state, d_state_supp,
                 use_gate=True):
        super().__init__()
        out_dim = in_dim // 2
        self.upsample = PatchExpand2D(in_dim)
        self.gate = SkipAttentionGate(skip_dim, out_dim) if use_gate else None
        self.concat_proj = nn.Linear(out_dim + skip_dim, out_dim, bias=False)
        self.blocks = nn.ModuleList(
            VSSBlock(out_dim, d_state, d_state_supp) for _ in range(depth))

    def forward(self, x, skip):
        x = self.upsample(x)
        if self.gate is not None:
            skip = self.gate(skip, x)
        x = self.concat_proj(torch.cat([x, skip], dim=-1))
        for blk in self.blocks:
            x = blk(x)
        return x


class AuxHead(nn.Module):
    def __init__(self, dim, num_classes, scale):
        super().__init__()
        self.proj = nn.Conv2d(dim, num_classes, 1)
        self.scale = scale

    def forward(self, x):
        x = self.proj(x.permute(0, 3, 1, 2).contiguous())
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                             align_corners=False)


# =========================================================================== #
#  LoRA (Table 6 baseline)
# =========================================================================== #

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


# =========================================================================== #
#  PEFT-UMamba
# =========================================================================== #

class PEFTUMamba(nn.Module):
    def __init__(self, num_classes, depths=(2, 2, 9, 2),
                 dims=(96, 192, 384, 768), d_state=16, d_state_supp=4,
                 decoder_depth=2, use_scam=True, scam_hidden=64,
                 use_skip_gate=True, scam_ordered=False):
        super().__init__()
        self.num_classes = num_classes
        self.dims = dims
        self.scam_ordered = scam_ordered

        self.encoder = VMambaEncoder(depths, dims, d_state, d_state_supp)
        self.scam = SCAM(dims[-1], scam_hidden) if use_scam else None

        self.decoder = nn.ModuleList(
            DecoderLayer(dims[i], dims[i - 1], decoder_depth, d_state,
                         d_state_supp, use_gate=use_skip_gate)
            for i in (3, 2, 1))                       # 768→384→192→96
        self.aux_heads = nn.ModuleList([              # deepest → shallowest
            AuxHead(dims[2], num_classes, 16),
            AuxHead(dims[1], num_classes, 8),
            AuxHead(dims[0], num_classes, 4),
        ])
        self.final_expand = FinalPatchExpand2D(dims[0])
        self.head = nn.Linear(dims[0], num_classes)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        f1, f2, f3, f4 = self.encoder(x)
        if self.scam is not None:
            f4 = self.scam(f4, ordered=self.scam_ordered)
        aux, h = [], f4
        for i, (layer, skip) in enumerate(zip(self.decoder, (f3, f2, f1))):
            h = layer(h, skip)
            if return_aux:
                aux.append(self.aux_heads[i](h))
        logits = self.head(self.final_expand(h)).permute(0, 3, 1, 2)
        return (logits, aux) if return_aux else logits

    # ── PEFT helpers ─────────────────────────────────────────────────────────
    def freeze_encoder(self):
        """All pretrained encoder weights frozen; only *_supp params train."""
        for name, p in self.encoder.named_parameters():
            p.requires_grad_("_supp" in name)

    def encoder_ss2d(self) -> List[SS2D]:
        return [m for m in self.encoder.modules() if isinstance(m, SS2D)]

    def enable_lora(self, rank=4, alpha=8.0, targets=("in_proj", "out_proj")):
        for ss2d in self.encoder_ss2d():
            for t in targets:
                setattr(ss2d, t, LoRALinear(getattr(ss2d, t), rank, alpha))

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_report(self) -> Dict[str, float]:
        def n(mod, only_trainable=True):
            return sum(p.numel() for p in mod.parameters()
                       if p.requires_grad or not only_trainable) / 1e6
        supp = sum(p.numel() for nm, p in self.named_parameters()
                   if "_supp" in nm and p.requires_grad) / 1e6
        enc_supp = sum(p.numel() for nm, p in self.encoder.named_parameters()
                       if "_supp" in nm and p.requires_grad) / 1e6
        return {
            "total": self.total_param_count() / 1e6,
            "trainable": self.trainable_param_count() / 1e6,
            "encoder_trainable": n(self.encoder),
            "supp_all": supp,
            "supp_encoder": enc_supp,
            "scam": n(self.scam) if self.scam is not None else 0.0,
            "decoder": n(self.decoder),
        }


# =========================================================================== #
#  Pretrained weight loading — verified
# =========================================================================== #

_KEY_RULES = [
    (r"^module\.", ""),
    (r"^backbone\.", ""),
    (r"^encoder\.", ""),
    (r"\.op\.", ".self_attention."),                 # later VMamba naming
    (r"(\.blocks\.\d+)\.norm\.", r"\1.ln_1."),       # later VMamba naming
]


def _remap(key: str) -> str:
    for pat, rep in _KEY_RULES:
        key = re.sub(pat, rep, key)
    return key


def load_vmamba_pretrained(encoder: VMambaEncoder, ckpt_path: str) -> int:
    """
    Load ImageNet VMamba-Tiny weights into the encoder.
    Every frozen encoder tensor must load with a matching shape, otherwise a
    RuntimeError is raised (this prevents silently training on a random
    encoder).  Supplementary (*_supp) parameters are not in the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {_remap(k): v for k, v in state.items()}

    target = encoder.state_dict()
    required = [k for k in target if "_supp" not in k]
    to_load, wrong_shape = {}, []
    for k in required:
        if k in state:
            if state[k].shape == target[k].shape:
                to_load[k] = state[k]
            else:
                wrong_shape.append(
                    f"{k}: ckpt {tuple(state[k].shape)} vs model {tuple(target[k].shape)}")
    missing = [k for k in required if k not in to_load]

    if missing:
        msg = [f"[Pretrained] {len(missing)}/{len(required)} encoder tensors "
               f"could not be loaded from {ckpt_path}."]
        if wrong_shape:
            msg.append("Shape mismatches (first 10):")
            msg += ["  " + s for s in wrong_shape[:10]]
        absent = [k for k in missing if k not in state][:10]
        if absent:
            msg.append("Not found in checkpoint (first 10):")
            msg += ["  " + k for k in absent]
            msg.append("Checkpoint keys (first 10): " +
                       ", ".join(list(state)[:10]))
        msg.append("This model expects the ORIGINAL VMamba-Tiny release "
                   "(K=16, depths 2-2-9-2, no MLP).")
        raise RuntimeError("\n".join(msg))

    encoder.load_state_dict(to_load, strict=False)
    print(f"[Pretrained] loaded {len(to_load)}/{len(required)} encoder "
          f"tensors from {ckpt_path}")
    return len(to_load)


# =========================================================================== #
#  Builder
# =========================================================================== #

ORDERED_DATASETS = {"dataset702", "dataset701"}   # volumetric: ordered slices


def build_model(cfg) -> PEFTUMamba:
    """
    Reads (with defaults):
      cfg.model.num_classes, depths, feat_dims, pretrained_path, freeze_encoder
      cfg.model.ssm.d_state                     (16)
      cfg.model.peft.supp_state_dim             (K′, 4)
      cfg.model.peft.use_supplementary_scan     (True)
      cfg.model.peft.supp_init                  ("neighbourhood" | "s4d")
      cfg.model.peft.no_skip_gate               (False)
      cfg.model.peft.use_lora / lora_rank / lora_alpha
      cfg.model.use_scam                        (True)
      cfg.model.scam_hidden                     (64)
      cfg.model.decoder_depth                   (2)
      cfg.data.ordered                          (SCAM: batch = one ordered sequence)
    """
    mc, pc = cfg.model, cfg.model.peft
    use_supp = getattr(pc, "use_supplementary_scan", True)
    k_prime = pc.supp_state_dim if use_supp else 0
    dataset = getattr(cfg.data, "dataset", "")

    model = PEFTUMamba(
        num_classes=mc.num_classes,
        depths=tuple(getattr(mc, "depths", (2, 2, 9, 2))),
        dims=tuple(getattr(mc, "feat_dims", (96, 192, 384, 768))),
        d_state=mc.ssm.d_state,
        d_state_supp=k_prime,
        decoder_depth=getattr(mc, "decoder_depth", 2),
        use_scam=getattr(mc, "use_scam", True),
        scam_hidden=getattr(mc, "scam_hidden", 64),
        use_skip_gate=not getattr(pc, "no_skip_gate", False),
        scam_ordered=(getattr(cfg.data, "ordered", dataset in ORDERED_DATASETS)
                      and not getattr(mc, "scam_unordered", False)),
    )

    ckpt_path = getattr(mc, "pretrained_path", None)
    if ckpt_path:
        load_vmamba_pretrained(model.encoder, ckpt_path)
    else:
        print("[Pretrained] WARNING: no checkpoint given — encoder is random")

    # Neighbourhood Initialisation needs the loaded A_frz → run after loading
    init = getattr(pc, "supp_init", "neighbourhood")
    for m in model.modules():
        if isinstance(m, SS2D) and m.d_state_supp > 0:
            m.reset_supplementary(init)

    if getattr(mc, "freeze_encoder", True):
        model.freeze_encoder()
    if getattr(pc, "use_lora", False):
        model.enable_lora(rank=pc.lora_rank, alpha=pc.lora_alpha)

    r = model.param_report()
    print(f"[Model] total={r['total']:.2f}M  trainable={r['trainable']:.3f}M  "
          f"(supp all={r['supp_all']:.3f}M, supp encoder={r['supp_encoder']:.3f}M, "
          f"SCAM={r['scam']:.3f}M, decoder={r['decoder']:.3f}M)  "
          f"K'={k_prime}  init={init}  SCAM ordered={model.scam_ordered}")
    return model
