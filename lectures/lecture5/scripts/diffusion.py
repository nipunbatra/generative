"""Everything Lecture 5 trains or samples: the schedule, the networks, the training step and the samplers.

The MNIST networks (TinyUNet, ConvAE, LatentDenoiser, Classifier) are copied unchanged from the
Generative Modelling II rebuild (dl-teaching/generative-modelling-ii/scripts/train_models.py), so its
validated checkpoints load as they are. ToyDenoiser is new: the 2-D smiley model, small enough to run
in the browser.

Conventions everywhere: linear beta from 1e-4 to 0.02, T = 1000, one-based t, alpha_bar[0] = 1,
images scaled to [-1, 1] (paper = -1, ink = +1), the network predicts the noise epsilon.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

T = 1000


# ------------------------------------------------------------------ schedules
def schedule(kind="linear", device="cpu", dtype=torch.float32):
    """beta, alpha, alpha_bar and the posterior variance beta-tilde, indexed 0..T (index 0 is the clean image)."""
    if kind == "linear":
        beta = torch.zeros(T + 1, dtype=torch.float64)
        beta[1:] = torch.linspace(1e-4, 2e-2, T, dtype=torch.float64)
        ab = torch.cumprod(1 - beta, 0)
    elif kind == "cosine":  # Nichol & Dhariwal (2021): alpha_bar follows a squared cosine
        s = 0.008
        u = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]
        beta = torch.zeros(T + 1, dtype=torch.float64)
        beta[1:] = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
        ab = torch.cumprod(1 - beta, 0)
    else:
        raise ValueError(kind)
    alpha = 1 - beta
    var = torch.zeros_like(beta)
    var[1:] = beta[1:] * (1 - ab[:-1]) / (1 - ab[1:])
    return [v.to(device=device, dtype=dtype) for v in (beta, alpha, ab, var)]


def time_features(t, dim=64):
    """Sinusoidal features of the step t, as for token positions in a transformer."""
    freq = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2 - 1))
    ang = t.float()[:, None] * freq
    return torch.cat([ang.sin(), ang.cos()], 1)


# ------------------------------------------------------------------ the 2-D smiley denoiser
class ToyDenoiser(nn.Module):
    """eps_hat(x, t[, c]) for 2-D points: [x, y, 16 time features (, one-hot part)] -> 3 x 128 SiLU -> 2.

    n_parts > 0 adds a one-hot label with n_parts + 1 entries; the last entry means "no label".
    """

    def __init__(self, n_parts=0, width=128, tdim=16):
        super().__init__()
        self.n_parts, self.tdim = n_parts, tdim
        din = 2 + tdim + (n_parts + 1 if n_parts else 0)
        self.net = nn.Sequential(nn.Linear(din, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x, t, c=None):
        parts = [x, time_features(t, self.tdim)]
        if self.n_parts:
            parts.append(F.one_hot(c, self.n_parts + 1).float())
        return self.net(torch.cat(parts, 1))


# ------------------------------------------------------------------ the MNIST denoiser (unchanged from GM-II)
class TimeBlock(nn.Module):
    def __init__(self, cin, cout, temb=64):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.time = nn.Linear(temb, cout)          # project the time embedding to this block's channels
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, h):
        y = F.silu(self.norm1(self.conv1(x))) + self.time(h)[:, :, None, None]   # broadcast over H x W
        return F.silu(self.norm2(self.conv2(y)) + self.skip(x))


class TinyUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28 with channels 32, 64, 128, 64, 32. n_classes > 0 adds a label (10 = no label)."""

    def __init__(self, n_classes=0):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.label = nn.Embedding(n_classes + 1, 64) if n_classes else None
        self.encoder1 = TimeBlock(1, 32)
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.encoder2 = TimeBlock(64, 64)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.middle = TimeBlock(128, 128)
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder1 = TimeBlock(128, 64)
        self.up2 = nn.Conv2d(64, 32, 3, padding=1)
        self.decoder2 = TimeBlock(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x, t, c=None):
        h = self.time_mlp(time_features(t))
        if self.label is not None:
            h = h + self.label(c)
        e1 = self.encoder1(x, h)
        e2 = self.encoder2(self.down1(e1), h)
        m = self.middle(self.down2(e2), h)
        u1 = self.up1(F.interpolate(m, size=e2.shape[-2:], mode="nearest"))
        d1 = self.decoder1(torch.cat([u1, e2], 1), h)
        u2 = self.up2(F.interpolate(d1, size=e1.shape[-2:], mode="nearest"))
        d2 = self.decoder2(torch.cat([u2, e1], 1), h)
        return self.head(d2)


class ConvAE(nn.Module):
    """1x28x28 image <-> 4x7x7 spatial latent (196 numbers, 4x fewer than 784)."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Conv2d(64, 4, 3, 1, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1), nn.SiLU(), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 1, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        return self.dec(self.enc(x))


class LatentDenoiser(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.inp = nn.Conv2d(4, ch, 3, padding=1)
        self.blocks = nn.ModuleList([TimeBlock(ch, ch) for _ in range(4)])
        self.out = nn.Conv2d(ch, 4, 3, padding=1)

    def forward(self, z, t):
        h = self.time_mlp(time_features(t))
        y = self.inp(z)
        for b in self.blocks:
            y = b(y, h)
        return self.out(y)


class Classifier(nn.Module):
    """A plain MNIST classifier, used only to measure what guided samples look like."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 49, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ training: the step on the code slide
def training_step(model, x0, ab, opt, c=None, null=None, p_drop=0.1):
    """One DDPM update. With labels c, hide each label with probability p_drop (classifier-free guidance)."""
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,), device=x0.device)       # a random noise level per example
    eps = torch.randn_like(x0)                                # the noise we add: also the target
    a = ab[t].reshape(B, *([1] * (x0.ndim - 1)))
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps                 # the shortcut: jump straight to step t
    if c is not None:
        c = torch.where(torch.rand(B, device=x0.device) < p_drop, torch.full_like(c, null), c)
        loss = F.mse_loss(model(xt, t, c), eps)
    else:
        loss = F.mse_loss(model(xt, t), eps)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


# ------------------------------------------------------------------ sampling
def x0_from_eps(xt, eps, a):
    """Undo the shortcut: x0_hat = (x_t - sqrt(1 - alpha_bar) eps_hat) / sqrt(alpha_bar)."""
    return (xt - (1 - a).sqrt() * eps) / a.sqrt()


@torch.no_grad()
def ddpm_step(eps, xt, t, sched, z=None, sigma="posterior"):
    """One reverse step x_t -> x_{t-1} from the predicted noise; z is the fresh noise (none at t = 1)."""
    beta, alpha, ab, var = sched
    mean = (xt - beta[t] / (1 - ab[t]).sqrt() * eps) / alpha[t].sqrt()
    if t == 1 or sigma is None:
        return mean
    s = var[t].sqrt() if sigma == "posterior" else beta[t].sqrt()
    return mean + s * (torch.randn_like(xt) if z is None else z)


@torch.no_grad()
def ddim_step(eps, xt, t, t_next, sched):
    """Deterministic DDIM: jump to the clean guess, then re-noise it to the next level with the same eps_hat."""
    ab = sched[2]
    x0 = x0_from_eps(xt, eps, ab[t])
    return ab[t_next].sqrt() * x0 + (1 - ab[t_next]).sqrt() * eps


def ddim_times(steps):
    """steps evenly spaced noise levels from T down to 1, then the clean level 0."""
    ts = torch.linspace(T, 1, steps).round().long().tolist()
    return ts + [0]
