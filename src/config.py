import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
N_EPOCHS = 10
NUM_WORKERS = 2
SAVE_MODEL = False
LOAD_MODEL = False