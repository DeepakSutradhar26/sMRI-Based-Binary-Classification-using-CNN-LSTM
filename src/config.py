import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 10
NUM_WORKERS = 2
SAVE_MODEL = False
LOAD_MODEL = False