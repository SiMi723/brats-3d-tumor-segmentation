from pathlib import Path

from torch.utils.data import DataLoader

from dataset.brats_dataset import BraTSDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.txt"
)


dataset = BraTSDataset(TRAIN_FILE)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)


images, masks = next(iter(loader))


print("Batch loaded successfully.")

print("\nImages:")
print("Shape:", images.shape)
print("Dtype:", images.dtype)

print("\nMasks:")
print("Shape:", masks.shape)
print("Dtype:", masks.dtype)

print("\nMask labels:")
print(masks.unique())