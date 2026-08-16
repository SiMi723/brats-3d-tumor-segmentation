import torch

from models.unet3d import UNet3D


model = UNet3D(
    in_channels=4,
    num_classes=4
)

x = torch.randn(
    1,
    4,
    96,
    96,
    96
)

with torch.no_grad():

    output = model(x)


print("Model forward pass successful.")

print("\nInput shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nNumber of parameters:")
print(
    sum(
        p.numel()
        for p in model.parameters()
    )
)