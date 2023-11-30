import constants
import wandb

def init():
  # start a new wandb run to track this script
  if constants.WANDB_ON:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Yolo",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": constants.LEARNING_RATE,
        "batch_size": constants.BATCH_SIZE,
        "B": constants.B,
        "C": constants.C,
        "epochs": constants.NUM_OF_EPOCHS,
        "image_height" : constants.IMAGE_HEIGHT,
        "image_width" : constants.IMAGE_WIDTH,
        "mask_threshhold" : constants.MASK_THRESHOLD,
        "confidence_threshhold" : constants.CONFIDENCE_THRESHOLD,
        "margin" : constants.MARGIN
        }
    )

def log(metrics):
  if constants.WANDB_ON:
    wandb.log(metrics)

def finish():
  if constants.WANDB_ON:
    wandb.finish()