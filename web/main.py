from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
import torch
import sys
sys.path.append("C:\\GAN")
from models.generator import Generator
import io
from PIL import Image

app = FastAPI()

generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load("C:/GAN/checkpoints/generator.pth"))
generator.eval()


@app.get("/")
async def home(request: Request):
    with open("c:/GAN/web/templates/index.html", "r") as file:
        html = file.read()
    return HTMLResponse(content=html, status_code=200)


@app.get("/generate_face")
def generate_face(name: str):
    torch.manual_seed(abs(hash(name)) % (2**32))  # Generate seed from the name
    z = torch.randn(1, 100)
    with torch.no_grad():
        fake_img = generator(z)[0]
    fake_img = (fake_img.clamp(-1, 1) + 1) / 2 * 255  # Scale to [0, 255]
    img_pil = Image.fromarray(fake_img.permute(1, 2, 0).byte().numpy())
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="PNG")
    return Response(content=img_bytes.getvalue(), media_type="image/png")
