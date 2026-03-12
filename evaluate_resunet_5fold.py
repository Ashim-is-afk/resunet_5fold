import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset

from train_resunet_5fold import (
    DEVICE,
    build_model,
    make_case_dicts,
    get_transforms,
    validate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main(args):
    output_dir = Path(args.output_dir)
    num_workers = args.num_workers

    all_metrics = []

    for fold in range(5):
        fold_dir = output_dir / f"fold_{fold}"
        val_csv = output_dir / f"fold_{fold}_val.csv"
        best_model_path = fold_dir / "best_model.pt"

        if not val_csv.exists() or not best_model_path.exists():
            log.warning(f"Skipping fold {fold}: missing files")
            continue

        ckpt = torch.load(best_model_path, map_location=DEVICE)
        ckpt_config = ckpt.get("config", {})

        patch_size = tuple(ckpt_config.get("patch_size", args.patch_size))
        spacing = tuple(ckpt_config.get("spacing", args.spacing))

        log.info(
            f"Fold {fold}: using patch_size={patch_size}, spacing={spacing}, "
            f"checkpoint={best_model_path}"
        )

        val_df = pd.read_csv(val_csv)
        val_cases = make_case_dicts(val_df)

        _, val_transforms = get_transforms(
            patch_size=patch_size,
            spacing=spacing,
            num_samples=1,
        )

        val_ds = Dataset(data=val_cases, transform=val_transforms)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(DEVICE.type == "cuda"),
        )

        model = build_model().to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        metrics = validate(model, val_loader, roi_size=patch_size)
        metrics["fold"] = fold
        all_metrics.append(metrics)

        log.info(f"Fold {fold}: {metrics}")

    if not all_metrics:
        raise RuntimeError("No folds evaluated")

    df = pd.DataFrame(all_metrics)
    df.to_csv(output_dir / "crossval_metrics.csv", index=False)

    summary = {}
    for col in ["dice", "iou", "hd95", "precision", "recall", "specificity", "accuracy", "f1"]:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    with open(output_dir / "crossval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="resunet_output",
        help="Folder containing fold CSVs and checkpoints",
    )
    parser.add_argument(
        "--patch_size",
        nargs=3,
        type=int,
        default=[96, 96, 96],
        help="Fallback patch size if checkpoint config is missing",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.5, 1.5, 3.0],
        help="Fallback spacing if checkpoint config is missing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    main(args)