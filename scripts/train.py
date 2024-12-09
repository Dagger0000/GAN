import torch
import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append("C:\\Style GAN")
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.generator import Generator
from models.discriminator import Discriminator

# Parameters
latent_dim = 100
epochs = 5000
batch_size = 64
learning_rate = 0.0002

# Data loading and preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ]
)

dataset = datasets.ImageFolder(root="C:/Style GAN/data/processed", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
generator = Generator(latent_dim)
discriminator = Discriminator()

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
)

# Training loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # 1. Train Discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(real_images.size(0), 1)

        real_images = real_images
        outputs = discriminator(real_images)
        real_loss = criterion(outputs, real_labels)

        z = torch.randn(real_images.size(0), latent_dim)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        fake_loss = criterion(outputs, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # 2. Train Generator
        generator.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Save model checkpoints
    if epoch % 100 == 0:
        torch.save(generator.state_dict(), "C:/Style GAN/checkpoints/generator.pth")
        torch.save(
            discriminator.state_dict(), "C:/Style GAN/checkpoints/discriminator.pth"
        )
