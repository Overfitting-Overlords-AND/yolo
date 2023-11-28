import torch
import torchvision.transforms.functional as TF

# Example tensor image
# Assume image_tensor is your existing image tensor, e.g., of shape [C, H, W]
image_tensor = torch.randn(1, 28, 28)  # Replace this with your actual image tensor

# Specify new size
new_size = (180, 180)  # for example, (height, width)

# Resize the image tensor
resized_image_tensor = TF.resize(image_tensor, new_size)

# resized_image_tensor is now the resized image tensor
