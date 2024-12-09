import torch


def name_to_latent_vector(name: str, latent_dim: int = 100):
    torch.manual_seed(abs(hash(name)) % (2**32))
    latent_vector = torch.randn(1, latent_dim)
    return latent_vector
