import torch
import sys

sys.path.append("C:\\GAN")
from models.generator import Generator
from utils.name_to_latent import name_to_latent_vector
import PIL.Image as Image

# Load the generator
generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load("C:/GAN/checkpoints/generator.pth"))
generator.eval()

name = input("Enter a name: ")
z = name_to_latent_vector(name)
with torch.no_grad():
    generated_img = generator(z)[0]

generated_img = (generated_img.clamp(-1, 1) + 1) / 2 * 255
img_pil = Image.fromarray(generated_img.permute(1, 2, 0).byte().numpy())
img_pil.save(f"{name}.png")
