import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.brats_dataset import BraTSDataset


TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.txt"
)


dataset = BraTSDataset(TRAIN_FILE)

print("Dataset size:")
print(len(dataset))

print("\nLoading first sample...")

image, mask = dataset[0]

print("\nImage:")
print("Shape:", image.shape)
print("Dtype:", image.dtype)

print("\nMask:")
print("Shape:", mask.shape)
print("Dtype:", mask.dtype)

print("\nMask labels:")
print(mask.unique())