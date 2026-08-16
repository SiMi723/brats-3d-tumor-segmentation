import torch

from losses.segmentation_loss import CombinedLoss


loss_function = CombinedLoss()


logits = torch.randn(
    2,
    4,
    96,
    96,
    96
)

targets = torch.randint(
    0,
    4,
    (
        2,
        96,
        96,
        96
    )
)


loss = loss_function(
    logits,
    targets
)


print("Loss calculation successful.")

print("\nLogits shape:")
print(logits.shape)

print("\nTarget shape:")
print(targets.shape)

print("\nLoss:")
print(loss.item())