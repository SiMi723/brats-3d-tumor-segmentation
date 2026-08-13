from pathlib import Path

from nifti_io import load_case
from cropping import crop_case
from normalization import normalize_case


# Find one BraTS training case
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
    / "BraTS20_Training_001"
)


print("Loading case...")

volumes, affine = load_case(CASE_DIR)

print("\nOriginal shapes:")

for name, volume in volumes.items():
    print(f"{name}: {volume.shape}")


# --------------------------------------------------
# Crop
# --------------------------------------------------

cropped_volumes = crop_case(volumes)

print("\nShapes after cropping:")

for name, volume in cropped_volumes.items():
    print(f"{name}: {volume.shape}")


# --------------------------------------------------
# Normalize MRI modalities
# --------------------------------------------------

normalized_volumes = normalize_case(cropped_volumes)

print("\nStatistics after normalization:")

for name, volume in normalized_volumes.items():

    if name == "seg":
        print(
            f"{name}: "
            f"min={volume.min()}, "
            f"max={volume.max()}"
        )

    else:
        mask = volume != 0

        print(
            f"{name}: "
            f"mean={volume[mask].mean():.4f}, "
            f"std={volume[mask].std():.4f}"
        )


# --------------------------------------------------
# Check segmentation labels
# --------------------------------------------------

seg = normalized_volumes["seg"]

print("\nSegmentation labels:")

print(sorted(set(seg.flatten())))

print("\nPreprocessing test completed.")