import torch
from torch.optim import Adam

from dataset.brats_dataset import BraTSDataset
from models.unet3d import UNet3D
from losses.segmentation_loss import CombinedLoss

from torch.utils.data import DataLoader

from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.txt"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = BraTSDataset(
    TRAIN_FILE
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet3D(
    in_channels=4,
    num_classes=4
)

model = model.to(device)


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = CombinedLoss()

optimizer = Adam(
    model.parameters(),
    lr=1e-4
)


# --------------------------------------------------
# Get one real batch
# --------------------------------------------------

images, masks = next(iter(loader))

images = images.to(device)
masks = masks.to(device)


print("\nInput:")
print(images.shape)

print("\nTarget:")
print(masks.shape)

print("\nTarget labels:")
print(masks.unique())


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

outputs = model(images)

print("\nOutput:")
print(outputs.shape)


# --------------------------------------------------
# Calculate loss
# --------------------------------------------------

loss = criterion(
    outputs,
    masks
)

print("\nLoss before backward:")
print(loss.item())


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

optimizer.zero_grad()

loss.backward()


# --------------------------------------------------
# Check gradients
# --------------------------------------------------

gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        gradient_found = True

        print(
            "\nGradient found in:",
            name
        )

        print(
            "Gradient mean:",
            parameter.grad.abs().mean().item()
        )

        break


# --------------------------------------------------
# Update model
# --------------------------------------------------

optimizer.step()


print("\nGradient calculation successful:")
print(gradient_found)

print("\nOptimizer step completed.")

print("\nSingle training step successful.")