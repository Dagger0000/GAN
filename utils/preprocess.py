"""from PIL import Image
import os

def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        with Image.open(img_path) as img:
            img = img.resize((64, 64))
            img.save(os.path.join(output_dir, img_name))"""

import os
from torchvision import transforms
from PIL import Image

# Define paths
raw_path = "C:\\Style GAN\\data\\raw"
processed_path = "C:/Style GAN/data/processed"

# Create the processed directory if it doesn't exist
os.makedirs(processed_path, exist_ok=True)

# Define the preprocessing transformations
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB (if necessary)
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip
    ]
)

# Process all images
for person_folder in os.listdir(raw_path):
    person_path = os.path.join(raw_path, person_folder)
    if os.path.isdir(person_path):
        processed_person_path = os.path.join(processed_path, person_folder)
        os.makedirs(processed_person_path, exist_ok=True)

        for image_file in os.listdir(person_path):
            if image_file.endswith((".jpg", ".png", ".jpeg")):
                try:
                    # Open and process the image
                    image_path = os.path.join(person_path, image_file)
                    image = Image.open(image_path)
                    image = transform(image)  # Apply transformations
                    save_path = os.path.join(processed_person_path, image_file)

                    # Convert back to PIL image to save
                    image.save(save_path)

                    print(f"Processed and saved: {save_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
