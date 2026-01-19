from torch.utils.data import DataLoader
from data_pipeline.dataset import MRIDataset
from sklearn.model_selection import train_test_split
import config
import os

import data_dowload

pateints = []
for label in ["HGG", "LGG"]:
    data_path = os.path.join(data_dowload.DATA_PATH, label)
    for p in os.listdir(data_path):
        pateints.append((label, p))

train_ids, test_ids = train_test_split(
    pateints,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_loader = DataLoader(
    MRIDataset(train_ids),
    batch_size = config.BATCH_SIZE,
    num_workers = config.NUM_WORKERS,
    shuffle = True,
    pin_memory = True,
)

val_loader = DataLoader(
    MRIDataset(test_ids),
    batch_size = config.BATCH_SIZE,
    num_workers = config.NUM_WORKERS,
    shuffle = True,
    pin_memory = True,
)