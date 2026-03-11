import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_resunet_5fold import (
    DEVICE,
    build_model,
    make_case_dicts,
    get_transforms,
    validate,
)
from monai.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("resunet_output")
PATCH_SIZE = (96, 96, 96)
SPACING = (1.5, 1.5, 3.0)
NUM_WORKERS = 0


def main():
    all_metrics = []

    for fold in range(5):
        fold_dir = OUTPUT_DIR / f"fold_{fold}"
        val_csv = OUTPUT_DIR / f"fold_{fold}_val.csv"
        best_model_path = fold_dir / "best_model.pt"

        if not val_csv.exists() or not best_model_path.exists():
            log.warning(f"Skipping fold {fold}: missing files")
            continue

        val_df = pd.read_csv(val_csv)
        val_cases = make_case_dicts(val_df)

        _, val_transforms = get_transforms(
            patch_size=PATCH_SIZE,
            spacing=SPACING,
            num_samples=1,
        )

        val_ds = Dataset(data=val_cases, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        ckpt = torch.load(best_model_path, map_location=DEVICE)
        model = build_model().to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        metrics = validate(model, val_loader, roi_size=PATCH_SIZE)
        metrics["fold"] = fold
        all_metrics.append(metrics)

        log.info(f"Fold {fold}: {metrics}")

    if not all_metrics:
        raise RuntimeError("No folds evaluated")

    df = pd.DataFrame(all_metrics)
    df.to_csv(OUTPUT_DIR / "crossval_metrics.csv", index=False)

    summary = {}
    for col in ["dice", "iou", "hd95", "precision", "recall", "specificity", "accuracy", "f1"]:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    with open(OUTPUT_DIR / "crossval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()