import torch
import torchvision
import torchvision.transforms.functional as TF
import random
import constants as c
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class DigitsDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.trn_ds = torchvision.datasets.MNIST(root="data", train=True,  download=True, transform=torchvision.transforms.ToTensor())

  def __getitem__(self, idx):
    image, digit_label = self.trn_ds.__getitem__(idx)
    new_size = random.randint(c.MINIMUM_DIGIT_SIZE, min(c.IMAGE_HEIGHT, c.IMAGE_WIDTH) - 2 * c.MARGIN)
    x1 = c.MARGIN + random.randint(0, c.IMAGE_WIDTH - 2 * c.MARGIN - new_size)
    y1 = c.MARGIN + random.randint(0, c.IMAGE_HEIGHT - 2 * c.MARGIN - new_size)
    resized_image = TF.resize(image, new_size)
    return_image = torch.zeros(c.IMAGE_HEIGHT, c.IMAGE_WIDTH)
    return_image[y1:y1+new_size, x1:x1+new_size] = resized_image
    return_image = return_image > c.WHITE_THRESHOLD
    xmin, ymin, xmax, ymax = torchvision.ops.masks_to_boxes(return_image.unsqueeze(0))[0]
    x, y, w, h = (xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin
    return return_image, digit_label, x, y, w, h

  def __len__(self):
    return self.trn_ds.__len__()

if __name__ == "__main__":
  dd = DigitsDataset()
  while True:
    image, digit_label, x, y, w, h = dd.__getitem__(random.randint(0,dd.__len__()))

    fig, ax = plt.subplots()
    ax.set_title(digit_label)
    ax.imshow(image.squeeze(0).numpy(), cmap='gray')
    ax.add_patch(Rectangle((x-w/2,y-h/2),width=w, height=h, edgecolor='Red', facecolor='none'))
    
    plt.axis('off')
    plt.show()
