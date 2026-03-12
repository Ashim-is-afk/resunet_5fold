import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)

from train_resunet_5fold import DEVICE, build_model

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NUM_WORKERS = 0 if sys.platform == "win32" else 4


def collect_cases(data_root: Path, only_without_mask: bool):
    cases = []

    for dataset_name in ["cptac", "ecpc"]:
        dataset_dir = data_root / dataset_name
        if not dataset_dir.exists():
            continue

        for patient_dir in sorted(dataset_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            image_path = patient_dir / "image.nii.gz"
            mask_path = patient_dir / "mask.nii.gz"

            if not image_path.exists():
                log.warning(f"SKIP {patient_dir}: missing image.nii.gz")
                continue

            if only_without_mask and mask_path.exists():
                log.info(f"SKIP {patient_dir.name}: mask.nii.gz exists")
                continue

            cases.append(
                {
                    "image": str(image_path),
                    "patient_id": patient_dir.name,
                    "case_id": patient_dir.name,
                    "dataset_type": dataset_name,
                }
            )

    return cases


def get_inference_transforms(spacing=(1.5, 1.5, 3.0)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=spacing,
            mode=("bilinear",),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])


@torch.no_grad()
def main(args):
    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    ckpt_config = ckpt.get("config", {})

    patch_size = tuple(args.patch_size) if args.patch_size is not None else tuple(
        ckpt_config.get("patch_size", [96, 96, 96])
    )
    spacing = tuple(args.spacing) if args.spacing is not None else tuple(
        ckpt_config.get("spacing", [1.5, 1.5, 3.0])
    )

    cases = collect_cases(data_root, only_without_mask=args.only_without_mask)

    if len(cases) == 0:
        raise RuntimeError("No valid inference cases found.")

    log.info(f"Device: {DEVICE}")
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"Cases found: {len(cases)}")
    log.info(f"Patch size: {patch_size}")
    log.info(f"Spacing: {spacing}")

    infer_transforms = get_inference_transforms(spacing=spacing)

    dataset = Dataset(data=cases, transform=infer_transforms)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=args.threshold),
        Invertd(
            keys="pred",
            transform=infer_transforms,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=False,
        ),
        SaveImaged(
            keys="pred",
            output_dir=str(pred_dir),
            output_postfix="pred",
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False,
            print_log=False,
        ),
    ])

    saved_rows = []

    for batch in loader:
        images = batch["image"].to(DEVICE)

        outputs = sliding_window_inference(
            inputs=images,
            roi_size=patch_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
        )

        batch["pred"] = outputs
        batch_list = decollate_batch(batch)

        for item in batch_list:
            patient_id = str(item["patient_id"])
            case_id = str(item["case_id"])
            dataset_type = str(item["dataset_type"])
            image_path = str(item["image_meta_dict"]["filename_or_obj"])

            saved_item = post_transforms(item)

            # SaveImaged writes the true output path into pred_meta_dict["filename_or_obj"]
            pred_meta = saved_item.get("pred_meta_dict", {})
            saved_path = pred_meta.get("filename_or_obj")

            saved_rows.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "dataset_type": dataset_type,
                    "image_path": image_path,
                    "prediction_path": str(saved_path) if saved_path is not None else "",
                }
            )

            log.info(f"Saved prediction for {patient_id}: {saved_path}")

    with open(output_dir / "inference_predictions.json", "w") as f:
        json.dump(saved_rows, f, indent=2)

    summary = {
        "n_cases": len(saved_rows),
        "checkpoint": str(checkpoint_path),
        "patch_size": list(patch_size),
        "spacing": list(spacing),
        "threshold": args.threshold,
        "overlap": args.overlap,
        "only_without_mask": args.only_without_mask,
        "device": str(DEVICE),
    }

    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--output_dir", default="resunet_inference_results")
    parser.add_argument("--only_without_mask", action="store_true")
    parser.add_argument("--patch_size", nargs=3, type=int, default=None)
    parser.add_argument("--spacing", nargs=3, type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)

    args = parser.parse_args()
    main(args)