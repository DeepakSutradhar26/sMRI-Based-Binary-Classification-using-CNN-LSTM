from scipy.ndimage import zoom

MID_PART = 64

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()

    data = (data - min_val) / (max_val - min_val + 1e-8)

    return data

def rescaled_data(data):
    D = data.shape[2]

    start = (D - MID_PART) // 2
    end = start + MID_PART

    middle_parts = data[:, :, start:end]

    target_size = (128, 128, MID_PART)
    zoom_factors = (target_size[0]/data.shape[0], target_size[1]/data.shape[1], 1)

    resized_data = zoom(middle_parts, zoom_factors, order=1)

    return resized_data