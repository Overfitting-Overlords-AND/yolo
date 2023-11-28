import torch
from torchvision import transforms
import random
from PIL import Image

# Example: Load a 28x28 grayscale image (you can replace this with your image loading code)
image = Image.open('path/to/your/28x28/image.png').convert('L')

# Randomly choose new dimensions
new_width = random.randint(28, 180)
new_height = random.randint(28, 180)

# Define the transformation - Resize and then convert to tensor
transform = transforms.Compose([
    transforms.Resize((new_height, new_width)),  # Resize the image
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
])

# Apply the transform to the image
resized_tensor_image = transform(image)

# resized_tensor_image is now a tensor of the randomly chosen size
