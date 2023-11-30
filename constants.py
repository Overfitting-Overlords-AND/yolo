MARGIN = 0
MINIMUM_DIGIT_SIZE = 100
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 700
MASK_THRESHOLD = 0.5
SR = 2 # rows of cells
SC = 7 # columns of cells
B = 1  # bounding boxes per cell
C = 10 # digit classifier

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
NUM_OF_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
