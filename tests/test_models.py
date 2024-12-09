import pytest
import torch
import sys

sys.path.append("C:\\Style GAN")
from models.generator import Generator
from models.discriminator import Discriminator

@pytest.fixture
def generator():
    latent_dim = 100
    return Generator(latent_dim)


@pytest.fixture
def discriminator():
    return Discriminator()


def test_generator_output_shape(generator):
    latent_dim = 100
    z = torch.randn(1, latent_dim)
    generated_img = generator(z)
    assert generated_img.shape == (1, 3, 64, 64)


def test_discriminator_output_shape(discriminator):
    img = torch.randn(1, 3, 64, 64)
    validity = discriminator(img)
    assert validity.shape == (1, 1)

# tests/test_models.py
def test_name_to_latent():
    """Test the name-to-latent vector conversion utility."""
    from utils.name_to_latent import name_to_latent_vector as name_to_latent

    name = "Test Name"
    latent_dim = 100
    latent_vector = name_to_latent(name, latent_dim)
    assert latent_vector.shape == (1, latent_dim), f"Expected (1, {latent_dim}), got {latent_vector.shape}"
    assert isinstance(latent_vector, torch.Tensor), "Latent vector should be a torch.Tensor"


def test_latent_determinism():
    """Test that the name-to-latent conversion is deterministic."""
    from utils.name_to_latent import name_to_latent_vector as name_to_latent

    name = "Deterministic Name"
    latent_dim = 100
    latent_vector1 = name_to_latent(name, latent_dim)
    latent_vector2 = name_to_latent(name, latent_dim)
    assert torch.equal(
        latent_vector1, latent_vector2
    ), "Latent vectors for the same name should be identical"

