from scipy.ndimage import zoom

MID_PART = 32

def rescaled_data(data):
    D = data.shape[2]

    start = (D - MID_PART) // 2
    end = start + MID_PART

    middle_parts = data[:, :, start:end]

    target_size = (128, 128, MID_PART)
    zoom_factors = (target_size[0]/data.shape[0], target_size[1]/data.shape[1], 1)

    resized_data = zoom(middle_parts, zoom_factors, order=1)

    return resized_data