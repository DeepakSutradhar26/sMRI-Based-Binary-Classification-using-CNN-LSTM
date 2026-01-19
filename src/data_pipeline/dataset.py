from torch.utils.data import Dataset
import nibabel as nib
import torch
import os

import preprocess.prerocessing as p

class MRIDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        label_name, patient_path = self.samples[index]
        nii_files = sorted(os.listdir(patient_path))

        volumes = []
        for nii in nii_files:
            nii_path = os.path.join(patient_path, nii)
            img = nib.load(nii_path)
            data = img.get_fdata()
            data = p.normalize_data(data)
            data = p.rescaled_data(data)
            volumes.append(torch.tensor(data, dtype = torch.float32))

        x = torch.stack(volumes) #(4, D, H, W)
        x = x.unsqueeze(1) # (4, 1, D, H, W)

        label = 1 if label_name == "HGG" else label = 0
        y = torch.tensor(label).unsqueeze(0)

        return x, y