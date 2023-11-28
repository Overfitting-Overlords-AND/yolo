import torch
from torchvision import transforms
from PIL import Image

# Load the image file
img_path = 'path/to/your/image.png'
image = Image.open(img_path)

# Convert the image to grayscale (if not already)
image = image.convert('L')

# Define a transform to convert the PIL image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
    # Add any other transforms you might need
])

# Apply the transform to the image
tensor_image = transform(image)

# tensor_image is now a PyTorch tensor ready for image processing
