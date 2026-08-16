import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):

        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        num_classes = logits.shape[1]

        probabilities = F.softmax(
            logits,
            dim=1
        )

        targets_one_hot = F.one_hot(
            targets,
            num_classes=num_classes
        )

        targets_one_hot = targets_one_hot.permute(
            0, 4, 1, 2, 3
        ).float()

        intersection = (
            probabilities * targets_one_hot
        ).sum(
            dim=(0, 2, 3, 4)
        )

        prediction_sum = (
            probabilities
        ).sum(
            dim=(0, 2, 3, 4)
        )

        target_sum = (
            targets_one_hot
        ).sum(
            dim=(0, 2, 3, 4)
        )

        dice = (
            2 * intersection + self.smooth
        ) / (
            prediction_sum
            + target_sum
            + self.smooth
        )

        dice_loss = 1 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.dice_loss = DiceLoss()

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets):

        ce = self.cross_entropy(
            logits,
            targets
        )

        dice = self.dice_loss(
            logits,
            targets
        )

        return ce + dice