import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import DigitsDataset
import utilities
import constants
from loss import YoloLoss

torch.manual_seed(123)

DEVICE = utilities.getDevice()

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for (image, label_matrix, digit_label, x, y, w, h) in loop:
        image, label_matrix = image.to(DEVICE), label_matrix.to(DEVICE)
        out = model(image)
        loss = loss_fn(out, label_matrix)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())
    
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(cell_rows=2, cell_columns=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE, weight_decay=constants.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    start_epoch = utilities.load_latest_checkpoint(model)

    train_dataset = DigitsDataset()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=constants.BATCH_SIZE,
        num_workers=constants.NUM_WORKERS,
        pin_memory=constants.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(start_epoch, constants.NUM_OF_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)
        utilities.save_checkpoint(model.state_dict(), f"./output/epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()