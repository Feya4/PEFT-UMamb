
#peft_baselines.py 
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    """h + up(GELU(down(h))), up zero-initialised (identity at start)."""
    def __init__(self, dim, bottleneck=16):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, h):
        return self.up(F.gelu(self.down(h)))

    def forward(self, h):
        return h + self.delta(h)


class LoRandAdapter(nn.Module):
    """Down/up projections are sums of α low-rank products P_i K Q_i with a
    kernel K shared by down and up (LoRand). Up path zero-initialised."""
    def __init__(self, dim, bottleneck=16, alpha=2, beta=8):
        super().__init__()
        self.K = nn.Parameter(torch.randn(alpha, beta, beta) * 0.02)
        self.Pd = nn.Parameter(torch.randn(alpha, dim, beta) * 0.02)
        self.Qd = nn.Parameter(torch.randn(alpha, beta, bottleneck) * 0.02)
        self.Pu = nn.Parameter(torch.randn(alpha, bottleneck, beta) * 0.02)
        self.Qu = nn.Parameter(torch.zeros(alpha, beta, dim))

    def forward(self, h):
        W_down = torch.einsum("adb,abc,ace->de", self.Pd, self.K, self.Qd)
        W_up = torch.einsum("adb,abc,ace->de", self.Pu, self.K, self.Qu)
        return h + F.gelu(h @ W_down.to(h.dtype)) @ W_up.to(h.dtype)


class MonaOp(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv3 = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.conv5 = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.conv7 = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        x = x + (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3.0
        return x + self.proj(x)


class MonaAdapter(nn.Module):
    """Scaled LayerNorm → down → multi-scale depthwise conv → GELU → up."""
    def __init__(self, dim, bottleneck=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(dim))
        self.down = nn.Linear(dim, bottleneck)
        self.op = MonaOp(bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):                            # [B,H,W,C]
        z = self.norm(h) * self.gamma + h * self.gammax
        z = self.down(z).permute(0, 3, 1, 2)
        z = self.op(z).permute(0, 2, 3, 1)
        return h + self.up(F.gelu(z))


class SublayerWithAdapter(nn.Module):
    def __init__(self, sublayer, adapter):
        super().__init__()
        self.sublayer = sublayer
        self.adapter = adapter

    def forward(self, x):
        return self.adapter(self.sublayer(x))


SERIAL = {"mona": MonaAdapter, "lorand": LoRandAdapter}


def apply_peft(encoder: nn.Module, method: str, bottleneck: int = 16) -> int:
    """Freezes the encoder, adds the baseline, returns #adaptation params."""
    for p in encoder.parameters():
        p.requires_grad_(False)

    if method in SERIAL:
        n_blocks = 0
        for layer in encoder.layers:
            for blk in layer.blocks:
                dim = blk.ln_1.normalized_shape[0]
                blk.self_attention = SublayerWithAdapter(
                    blk.self_attention, SERIAL[method](dim, bottleneck))
                n_blocks += 1
        assert n_blocks > 0, "no VSS blocks found in encoder.layers"

    elif method == "e3va":
        for layer in encoder.layers:
            dim = layer.blocks[0].ln_1.normalized_shape[0]
            layer.e3va = nn.ModuleList(
                BottleneckAdapter(dim, bottleneck) for _ in layer.blocks)

        def e3va_forward(x):
            with torch.no_grad():
                x = encoder.patch_embed(x)
            feats = []
            for layer in encoder.layers:
                side = 0.0
                for blk, ad in zip(layer.blocks, layer.e3va):
                    with torch.no_grad():
                        x = blk(x)
                    side = side + ad.delta(x.detach())
                feats.append(x + side)          # adapted skip feature
                if layer.downsample is not None:
                    with torch.no_grad():
                        x = layer.downsample(x)
            return feats

        encoder.forward = e3va_forward
    else:
        raise ValueError(f"unknown PEFT baseline: {method}")

    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)
