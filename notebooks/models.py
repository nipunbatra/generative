"""Teaching models. Adapted from the supplied course brief; raw logits for BCE."""
import torch
from torch import nn
from torch.nn import functional as F

class TinyVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Linear(784, 128)
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, 128)
        self.decoder_logits = nn.Linear(128, 784)

    def encode(self, x):
        h = F.relu(self.encoder(x.flatten(1)))
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.decoder_logits(F.relu(self.decoder_hidden(z)))

    def forward(self, x, eps=None):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        z = mu + std * eps
        return self.decode(z), mu, logvar

def vae_step(model, optimizer, x):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, mu, logvar = model(x)
    target = x.flatten(1)
    recon = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).sum(dim=1)
    kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=1)
    loss = (recon + kl).mean()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "recon": float(recon.detach().mean()),
        "kl": float(kl.detach().mean()),
    }

@torch.no_grad()
def sample_vae(model, count=16, sample_pixels=False):
    model.eval()
    device = next(model.parameters()).device
    latent_dim = model.mu_head.out_features
    z = torch.randn(count, latent_dim, device=device)
    probabilities = model.decode(z).sigmoid()
    output = torch.bernoulli(probabilities) if sample_pixels else probabilities
    return output.reshape(count, 1, 28, 28)


import torch
from torch import nn
from torch.nn import functional as F

def make_image_gan(latent_dim=32):
    generator = nn.Sequential(
        nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
        nn.Linear(128, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 784), nn.Sigmoid(),
    )
    discriminator = nn.Sequential(
        nn.Linear(784, 256), nn.LeakyReLU(0.2),
        nn.Linear(256, 128), nn.LeakyReLU(0.2),
        nn.Linear(128, 1),  # raw logits: no final sigmoid
    )
    return generator, discriminator

def gan_step(G, D, opt_G, opt_D, real, latent_dim=32):
    G.train()
    D.train()
    real = real.flatten(1)
    batch_size = real.shape[0]
    device = real.device

    # D phase: real=1, generated=0; fake.detach() isolates G.
    D.requires_grad_(True)
    opt_D.zero_grad(set_to_none=True)
    opt_G.zero_grad(set_to_none=True)
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)
    real_logits = D(real)
    fake_logits = D(fake.detach())
    loss_D = (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    )
    loss_D.backward()
    opt_D.step()

    # G phase: freeze D parameters, retain differentiation through D's input.
    opt_D.zero_grad(set_to_none=True)
    opt_G.zero_grad(set_to_none=True)
    D.requires_grad_(False)
    z = torch.randn(batch_size, latent_dim, device=device)
    fake = G(z)                  # fresh graph
    fake_logits = D(fake)        # no detach and no torch.no_grad()
    loss_G = F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )
    loss_G.backward()
    opt_G.step()
    D.requires_grad_(True)       # ready for the next D phase

    return {"D": float(loss_D.detach()), "G": float(loss_G.detach())}

@torch.no_grad()
def sample_gan(G, count=16, latent_dim=32):
    G.eval()
    device = next(G.parameters()).device
    z = torch.randn(count, latent_dim, device=device)
    return G(z).reshape(count, 1, 28, 28)


