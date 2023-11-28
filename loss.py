import torch
import torch.nn as nn
import constants as c

class YoloLoss(nn.Module):

    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, SR*SC*(C+B*5)) on input
        predictions = predictions.reshape(-1, c.SR, c.SC, c.C + c.B * 5)
        
        exists_box = target[..., 10].unsqueeze(3)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. 
        box_predictions = exists_box * predictions[..., 11:15]
        box_targets = exists_box * target[..., 11:15]

        # Take sqrt of width, height of boxes
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        object_loss = self.mse(
            torch.flatten(exists_box * predictions[..., 10:11]),
            torch.flatten(exists_box * target[..., 10:11]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 10:11], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 10:11], start_dim=1),
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :10], end_dim=-2,),
            torch.flatten(exists_box * target[..., :10], end_dim=-2,),
        )

        # ================== #
        #   TOTAL LOSS       #
        # ================== #

        loss = (
            self.lambda_coord * box_loss  
            + object_loss 
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
