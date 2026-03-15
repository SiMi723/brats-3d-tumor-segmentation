import numpy as np

def crop_brain(volume):
    
    non_zero = np.where(volume > 0)
    
    x_min, x_max = non_zero[0].min(), non_zero[0].max()
    y_min, y_max = non_zero[1].min(), non_zero[1].max()
    z_min, z_max = non_zero[2].min(), non_zero[2].max()

    cropped = volume[x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped