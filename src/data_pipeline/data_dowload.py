import kagglehub

DATA_PATH = kagglehub.dataset_download(
    "aryashah2k/brain-tumor-segmentation-brats-2019",
    limit=2,
)