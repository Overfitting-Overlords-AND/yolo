from PIL import Image
import torch
import constants as c
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from model import Yolov1
from dataset import DigitsDataset
import random

# # Load the image
# image_path = "./5.png"
# image = Image.open(image_path)

# # Convert the image to grayscale (if it's not already)
# image = image.convert("L")

# # Resize the image to 200x700 (if it's not already)
# image = image.resize((700, 200))

# # Convert the image to a PyTorch tensor
# tensor = torch.tensor(np.array(image))

# # Normalize the tensor so that its values are 1.0 or 0.0
# tensor = (tensor > 128).float()

dd = DigitsDataset()

# Example file path for the saved model
model_path = './output/epoch_3.pt'

# Load the model
model = Yolov1()
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

while True:
  # get an image
  tensor, label_matrix, digit_label, x, y, w, h = dd.__getitem__(random.randint(0,dd.__len__()))

  # Add a channel dimension at the first position
  # tensor = tensor.unsqueeze(0)  

  # Add a batch dimension at the first position
  tensor = tensor.unsqueeze(0)  

  # Pass the tensor to the model
  with torch.no_grad():
      output = model(tensor).reshape(2,7,15)

  # output now contains the model's output
      fig, ax = plt.subplots()
      ax.imshow(tensor.squeeze(0).squeeze(0).numpy(), cmap='gray')
      for cr in range(2):
          for cc in range(7):
            c, x, y, w, h = output[cr,cc,10:15]
            p = output[cr,cc,:10]
            x, y = 100 * (cc + x), 100 * (cr + y)
            w, h = w * 700, h * 200
            if c > .5:   
              ax.add_patch(Rectangle((x-w/2,y-h/2),width=w, height=h, edgecolor='Red', facecolor='none'))
              ax.set_title(torch.argmax(p).item())
      plt.axis('off')
      plt.show()
      print(output)
