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
    labels = []
    image = None
    label_matrix = None
    if torch.all(d1[0] * d2[0] == 0) and (d1[3]/100 != d2[3]/100 or d1[4]/100 != d2[4]/100):
      image = d1[0] + d2[0]
      label_matrix = d1[1] + d2[1]
      labels.append(d1[2:])   
      labels.append(d2[2:])   
    else:
      image = d1[0]
      label_matrix = d1[1]
      labels.append(d1[2:])
      labels.append(torch.zeros(5))
    return image, label_matrix, torch.cat(labels, dim=0)

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
  
def decode_label_matrix(self,label_matrix):
  non_zero_rows = []
  for i in range(label_matrix.size(2)):
    slice_ = label_matrix[:, :, i]
    non_zero_in_row = slice_.nonzero(as_tuple=True)[0].unique().tolist()
    non_zero_rows.append(non_zero_in_row)

  twoLabels = []
  for i in range(len(non_zero_rows)):
    labels = []
    labels.append(torch.argmax(non_zero_rows[i][:10]))
    labels.append(non_zero_rows[i][11])
    labels.append(non_zero_rows[i][12])
    labels.append(non_zero_rows[i][13])
    labels.append(non_zero_rows[i][14])
    twoLabels.append(labels)
  return twoLabels  

if __name__ == "__main__":
  dd = DigitsDataset()
  while True:
    image, label_matrix = dd.__getitem__(random.randint(0,dd.__len__()))
    
    dlm = decode_label_matrix(label_matrix)

    fig, ax = plt.subplots()
    ax.set_title(f"{dlm[0][0]} {dlm[1][0]}")
    ax.imshow(image.squeeze(0).numpy(), cmap='gray')

    # for item in labels:
    #   for label_matrix, digit_label, x, y, w, h in item:
    for i in range(len(dlm)):
      digit_label, x, y, w, h = dlm[i]
      ax.add_patch(Rectangle((x-w/2,y-h/2),width=w, height=h, edgecolor='Red', facecolor='none'))

    plt.axis('off')
    plt.show()
