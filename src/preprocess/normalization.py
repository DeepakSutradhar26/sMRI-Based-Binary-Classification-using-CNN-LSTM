import nibabel as nib
import glob
import os

curr_path = os.path.abspath(__file__)

root_path = os.path.abspath(os.path.join(curr_path, "..", "..", ".."))

def normalize_data(name):
    train_path = os.path.join(root_path, "data", name)
    norm_path  = os.path.join(root_path, "norm_data", name)

    if not os.path.exists(norm_path):
        os.makedirs(norm_path)

    folders = glob.glob(os.path.join(train_path, "*/"))

    for folder in folders:
        basename = os.path.basename(os.path.normpath(folder))
        files = glob.glob(os.path.join(folder, "*.nii*"))

        for file in files:
            normalize_file(file, os.path.join(norm_path, basename))

def normalize_file(input_path, output_path):
    basename = os.path.basename(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img = nib.load(input_path)
    data = img.get_fdata()

    min_val = data.min()
    max_val = data.max()

    data = (data - min_val) / (max_val - min_val + 1e-8)

    nifti_data = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(nifti_data, os.path.join(output_path, basename))

def normalize_train():
    name = "train"
    normalize_data(name)

def normalize_test():
    name = "test"
    normalize_data(name)