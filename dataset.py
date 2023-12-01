import torch
import torchvision
import torchvision.transforms.functional as TF
import random
import constants as c
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class DigitsDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.trn_ds = torchvision.datasets.MNIST(root="data", train=True,  download=True, transform=torchvision.transforms.ToTensor())
  
  def __getitem__(self, idx):
    d1 = self.get_one_digit(random.randint(0,self.__len__()) )
    d2 = self.get_one_digit(random.randint(0,self.__len__()) )
    image = None
    label_matrix = None
    if torch.all(d1[0] * d2[0] == 0) and (d1[3]/100 != d2[3]/100 or d1[4]/100 != d2[4]/100):
      image = d1[0] + d2[0]
      label_matrix = d1[1] + d2[1]
    else:
      image = d1[0]
      label_matrix = d1[1]
    return image, label_matrix

  def get_one_digit(self, idx):
    image, digit_label = self.trn_ds.__getitem__(idx)
    new_size = random.randint(c.MINIMUM_DIGIT_SIZE, min(c.IMAGE_HEIGHT, c.IMAGE_WIDTH) - 2 * c.MARGIN)
    x1 = c.MARGIN + random.randint(0, c.IMAGE_WIDTH - 2 * c.MARGIN - new_size)
    y1 = c.MARGIN + random.randint(0, c.IMAGE_HEIGHT - 2 * c.MARGIN - new_size)
    resized_image = TF.resize(image, new_size)
    return_image = torch.zeros(c.IMAGE_HEIGHT, c.IMAGE_WIDTH)
    return_image[y1:y1+new_size, x1:x1+new_size] = resized_image
    mask_image = return_image > c.MASK_THRESHOLD
    xmin, ymin, xmax, ymax = torchvision.ops.masks_to_boxes(mask_image.unsqueeze(0))[0]
    x, y, w, h = (xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin
    label_matrix = self.calculate_label_matrix(digit_label, x, y, w, h)
    return mask_image.float().unsqueeze(0), label_matrix, digit_label, x, y, w, h

  def __len__(self):
    return self.trn_ds.__len__()

  def calculate_label_matrix(self, digit_label, x, y, w, h):
    x, y = x / c.IMAGE_WIDTH, y / c.IMAGE_HEIGHT
    i, j = int(c.SR * y), int(c.SC * x)
    x_cell, y_cell = c.SC * x - j, c.SR * y - i
    w_cell, h_cell = c.SC * w / c.IMAGE_WIDTH, c.SR * h / c.IMAGE_HEIGHT
    label_matrix = torch.zeros((c.SR, c.SC, c.C + 5 * c.B))
    label_matrix[i, j, digit_label] = 1 # one hot encoding
    label_matrix[i, j, c.C] = 1 # cell contains centre of digit
    label_matrix[i, j, c.C+1:c.C+5] = torch.tensor([ x_cell, y_cell, w_cell, h_cell ])
    return label_matrix
  
def decode_label_matrix(label_matrix):
  output = []
  for i in range(4):
    for j in range(4):
      lm = label_matrix[i,j,:]
      if not torch.all(lm==0):
        digit_label = torch.argmax(lm[:10])
        c = lm[10]
        x = (lm[11]+j)*100
        y = (lm[12]+i)*100
        w = lm[13]*100
        h = lm[14]*100
        output.append((digit_label,c,x,y,w,h))
  return output

if __name__ == "__main__":
  dd = DigitsDataset()
  while True:
    image, label_matrix = dd.__getitem__(random.randint(0,dd.__len__()))
    
    dlm = decode_label_matrix(label_matrix)

    fig, ax = plt.subplots()

    # for item in labels:
    #   for label_matrix, digit_label, x, y, w, h in item:
    digit_labels = ""
    for i in range(len(dlm)):
      digit_label, _, x, y, w, h = dlm[i]
      digit_labels = f"{digit_labels} {str(digit_label)}"
      ax.add_patch(Rectangle((x-w/2,y-h/2),width=w, height=h, edgecolor='Red', facecolor='none'))
    ax.set_title(digit_labels)
    ax.imshow(image.squeeze(0).numpy(), cmap='gray')

    plt.axis('off')
    plt.show()
