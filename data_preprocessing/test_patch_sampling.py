from pathlib import Path

from nifti_io import load_case
from cropping import crop_case
from normalization import normalize_case
from patch_sampling import sample_patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
    / "BraTS20_Training_001"
)


# Load
volumes, affine = load_case(CASE_DIR)

# Crop
volumes = crop_case(volumes)

# Normalize MRI
volumes = normalize_case(volumes)

# Sample one patch
image_patch, mask_patch = sample_patch(volumes)


print("Patch sampling successful.")

print("\nImage patch shape:")
print(image_patch.shape)

print("\nMask patch shape:")
print(mask_patch.shape)

print("\nImage patch dtype:")
print(image_patch.dtype)

print("\nMask patch dtype:")
print(mask_patch.dtype)

print("\nMask labels:")
print(sorted(set(mask_patch.flatten())))