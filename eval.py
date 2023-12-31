from PIL import Image
import torch
import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from model import Yolov1
from dataset import DigitsDataset
from dataset import decode_label_matrix
import random
import utilities

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

# Load the model
model = Yolov1()

# Example file path for the saved model
utilities.load_latest_checkpoint(model)

# Set the model to evaluation mode
model.eval()

while True:
  # get an image
  tensor, label_matrix = dd.__getitem__(random.randint(0,dd.__len__()))
  dlm = decode_label_matrix(label_matrix)

  # Add a channel dimension at the first position
  tensor = tensor.unsqueeze(0)  

  # Add a batch dimension at the first position
  # tensor = tensor.unsqueeze(0)  

  # Pass the tensor to the model
  with torch.no_grad():
      output = model(tensor).reshape(4,4,15)
      fig, ax = plt.subplots()
      ax.imshow(tensor.squeeze(0).squeeze(0).numpy(), cmap='gray')
      for cr in range(4):
          for cc in range(4):
            c, x, y, w, h = output[cr,cc,10:15]
            p = output[cr,cc,:10]
            x, y = 100 * (cc + x), 100 * (cr + y)
            w, h = w * 100, h * 100
            if c > constants.CONFIDENCE_THRESHOLD:   
              ax.add_patch(Rectangle((x-w/2,y-h/2),width=w, height=h, edgecolor='Red', facecolor='none'))
              # ax.text(0, 0, torch.argmax(p).item())
              ax.set_title(torch.argmax(p).item())
      plt.axis('off')
      plt.show()
      print(output)