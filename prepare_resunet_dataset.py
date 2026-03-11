import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("prepare_resunet_dataset.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ========================= EDIT THESE =========================
DATA_ROOT = Path("data")
OUTPUT_DIR = Path("resunet_output")

N_SPLITS = 5
RANDOM_SEED = 42

MASK_MIN_VOXELS = 500
MASK_MAX_VOXELS = 1_500_000

# IMPORTANT:
# Change ECPC labels if your ECPC masks use something else
DATASET_LABEL_MAP = {
    "cptac": [3, 4],
    "ecpc": [1],
}
# =============================================================


def load_case(dataset_name: str, patient_dir: Path):
    image_path = patient_dir / "image.nii.gz"
    mask_path = patient_dir / "mask.nii.gz"

    if not image_path.exists() or not mask_path.exists():
        log.warning(f"SKIP {patient_dir}: missing image.nii.gz or mask.nii.gz")
        return None

    try:
        raw_mask = nib.load(str(mask_path)).get_fdata()
        binary_mask = np.isin(raw_mask, DATASET_LABEL_MAP[dataset_name]).astype(np.uint8)
        voxels = int(binary_mask.sum())
        unique_raw = np.unique(raw_mask).tolist()
    except Exception as e:
        log.warning(f"SKIP {patient_dir}: {e}")
        return None

    if voxels == 0:
        log.warning(f"SKIP {patient_dir.name}: empty mask after binarization")
        return None
    if voxels < MASK_MIN_VOXELS:
        log.warning(f"SKIP {patient_dir.name}: mask too small ({voxels})")
        return None
    if voxels > MASK_MAX_VOXELS:
        log.warning(f"SKIP {patient_dir.name}: mask too large ({voxels})")
        return None

    return {
        "patient_id": patient_dir.name,
        "dataset_type": dataset_name,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "mask_voxels": voxels,
        "mask_labels_raw": str(unique_raw),
    }


def main():
    cptac_dir = DATA_ROOT / "cptac"
    ecpc_dir = DATA_ROOT / "ecpc"

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = []

    if cptac_dir.exists():
        for patient_dir in sorted(cptac_dir.iterdir()):
            if patient_dir.is_dir():
                case = load_case("cptac", patient_dir)
                if case is not None:
                    cases.append(case)

    if ecpc_dir.exists():
        for patient_dir in sorted(ecpc_dir.iterdir()):
            if patient_dir.is_dir():
                case = load_case("ecpc", patient_dir)
                if case is not None:
                    cases.append(case)

    if len(cases) < N_SPLITS:
        raise RuntimeError(f"Only {len(cases)} valid cases found. Need at least {N_SPLITS}.")

    df = pd.DataFrame(cases)
    df["case_id"] = [f"ResUNet_{i+1:04d}" for i in range(len(df))]
    df.to_csv(OUTPUT_DIR / "case_split_log.csv", index=False)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    indices = np.arange(len(df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)

        train_df["fold"] = fold
        train_df["split"] = "train"
        val_df["fold"] = fold
        val_df["split"] = "val"

        train_df.to_csv(OUTPUT_DIR / f"fold_{fold}_train.csv", index=False)
        val_df.to_csv(OUTPUT_DIR / f"fold_{fold}_val.csv", index=False)

        log.info(f"Fold {fold}: train={len(train_df)} | val={len(val_df)}")

    summary = {
        "n_cases": int(len(df)),
        "n_splits": N_SPLITS,
        "dataset_counts": df["dataset_type"].value_counts().to_dict(),
        "label_map": DATASET_LABEL_MAP,
    }

    with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()