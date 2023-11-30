import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os
import re
import os
import re

def getDevice():
  return "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    # Extract the directory from the filename
    directory = os.path.dirname(filename)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, filename)

def find_latest_epoch_file(path='./output'):
    epoch_files = None
    if os.path.exists(path):
        epoch_files = [f for f in os.listdir(path) if re.match(r'epoch_\d+\.pt', f)]
    if epoch_files:
        # Extracting epoch numbers from the files and finding the max
        latest_epoch = max([int(f.split('_')[1].split('.')[0]) for f in epoch_files])
        return latest_epoch, f"{path}/epoch_{latest_epoch}.pt"
    else:
        return 0, None

# Function to load the latest epoch file if it exists
def load_latest_checkpoint(model, path='./output'):
    latest_epoch, latest_file = find_latest_epoch_file(path)
    if latest_file:
        print(f"Resuming training from epoch {latest_epoch+1}")
        model.load_state_dict(torch.load(latest_file, map_location=torch.device(getDevice())))
    else:
        print("No checkpoint found, starting from beginning")
    return latest_epoch