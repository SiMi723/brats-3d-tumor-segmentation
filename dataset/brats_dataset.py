from pathlib import Path

import torch
from torch.utils.data import Dataset

from data_preprocessing.nifti_io import load_case
from data_preprocessing.cropping import crop_case
from data_preprocessing.normalization import normalize_case
from data_preprocessing.patch_sampling import sample_patch
from data_preprocessing.label_mapping import remap_labels


class BraTSDataset(Dataset):

    def __init__(self, split_file):

        split_file = Path(split_file)

        self.project_root = split_file.resolve().parents[2]

        self.data_dir = (
            self.project_root
            / "data"
            / "BraTS2020_TrainingData"
            / "MICCAI_BraTS2020_TrainingData"
        )

        with open(split_file, "r") as f:

            self.case_ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

    def __len__(self):

        return len(self.case_ids)

    def __getitem__(self, index):

        case_id = self.case_ids[index]

        case_dir = self.data_dir / case_id

        # Load
        volumes, _ = load_case(case_dir)

        # Crop
        volumes = crop_case(volumes)

        # Normalize MRI
        volumes = normalize_case(volumes)

        # Sample a 3D patch
        image_patch, mask_patch = sample_patch(
            volumes
        )
        # Convert BraTS labels to consecutive training labels
        mask_patch = remap_labels(mask_patch)

        # Convert image to PyTorch tensor
        image_patch = torch.from_numpy(
            image_patch
        ).float()

        # Segmentation labels should be integers
        mask_patch = torch.from_numpy(
            mask_patch
        ).long()

        return image_patch, mask_patch