from pathlib import Path
import random


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# Split settings
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def get_valid_cases(data_dir):
    """Find cases that contain all required MRI files and a segmentation mask."""

    required = ["flair", "t1", "t1ce", "t2", "seg"]

    valid_cases = []
    skipped_cases = []

    for case_dir in sorted(data_dir.iterdir()):

        if not case_dir.is_dir():
            continue

        case_id = case_dir.name

        files_present = True

        for modality in required:

            file_path = case_dir / f"{case_id}_{modality}.nii"

            if not file_path.exists():
                files_present = False
                break

        if files_present:
            valid_cases.append(case_id)
        else:
            skipped_cases.append(case_id)

    return valid_cases, skipped_cases


def split_cases(case_ids):
    """Split cases into train, validation and test sets."""

    random.seed(SEED)

    case_ids = case_ids.copy()
    random.shuffle(case_ids)

    n = len(case_ids)

    train_end = int(TRAIN_RATIO * n)
    val_end = train_end + int(VAL_RATIO * n)

    train_cases = case_ids[:train_end]
    val_cases = case_ids[train_end:val_end]
    test_cases = case_ids[val_end:]

    return train_cases, val_cases, test_cases


def save_cases(case_ids, filename):

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    file_path = SPLIT_DIR / filename

    with open(file_path, "w") as f:
        for case_id in case_ids:
            f.write(case_id + "\n")

    return file_path


if __name__ == "__main__":

    print("Finding BraTS cases...")

    valid_cases, skipped_cases = get_valid_cases(DATA_DIR)

    print(f"Total valid cases: {len(valid_cases)}")

    if skipped_cases:
        print("\nCases skipped because of missing files:")

        for case in skipped_cases:
            print("-", case)

    train_cases, val_cases, test_cases = split_cases(valid_cases)

    print("\nDataset split:")
    print(f"Train: {len(train_cases)}")
    print(f"Validation: {len(val_cases)}")
    print(f"Test: {len(test_cases)}")

    save_cases(train_cases, "train.txt")
    save_cases(val_cases, "val.txt")
    save_cases(test_cases, "test.txt")

    print("\nSplit files saved to:")
    print(SPLIT_DIR)