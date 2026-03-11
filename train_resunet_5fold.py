import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False

NUM_WORKERS = 0 if sys.platform == "win32" else 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# IMPORTANT:
# Change ECPC labels if your ECPC masks use something else
DATASET_LABEL_MAP = {
    "cptac": [3, 4],
    "ecpc": [1],
}


class ConvertMaskByDatasetd(MapTransform):
    def __init__(self, keys, dataset_label_map):
        super().__init__(keys)
        self.dataset_label_map = dataset_label_map

    def __call__(self, data):
        d = dict(data)
        dataset_type = str(d["dataset_type"]).lower()

        if dataset_type not in self.dataset_label_map:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        fg_labels = self.dataset_label_map[dataset_type]

        for key in self.keys:
            mask = d[key]
            d[key] = np.isin(mask, fg_labels).astype(np.uint8)

        return d


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def build_model():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        act="PRELU",
        dropout=0.1,
    )


def get_transforms(patch_size=(96, 96, 96), spacing=(1.5, 1.5, 3.0), num_samples=4):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertMaskByDatasetd(keys=["label"], dataset_label_map=DATASET_LABEL_MAP),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=2,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertMaskByDatasetd(keys=["label"], dataset_label_map=DATASET_LABEL_MAP),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms


def make_case_dicts(df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        records.append({
            "image": row["image_path"],
            "label": row["mask_path"],
            "dataset_type": row["dataset_type"],
            "patient_id": row["patient_id"],
            "case_id": row["case_id"],
        })
    return records


def compute_binary_stats(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.logical_and(pred == 1, target == 1).sum()
    tn = np.logical_and(pred == 0, target == 0).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()

    eps = 1e-8
    return {
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "specificity": float(tn / (tn + fp + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "f1": float((2 * tp) / (2 * tp + fp + fn + eps)),
    }


@torch.no_grad()
def validate(model, loader, roi_size):
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    stats_all = {k: [] for k in ["precision", "recall", "specificity", "iou", "accuracy", "f1"]}

    for batch in tqdm(loader, desc="  Val", leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = sliding_window_inference(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

        pred_list = [post_pred(x) for x in decollate_batch(outputs)]
        label_list = [post_label(x) for x in decollate_batch(labels)]

        dice_metric(y_pred=pred_list, y=label_list)
        hd95_metric(y_pred=pred_list, y=label_list)

        for p, g in zip(pred_list, label_list):
            p_np = p.detach().cpu().numpy().astype(np.uint8).squeeze()
            g_np = g.detach().cpu().numpy().astype(np.uint8).squeeze()
            stats = compute_binary_stats(p_np, g_np)
            for k, v in stats.items():
                stats_all[k].append(v)

    results = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in stats_all.items()}
    results["dice"] = float(dice_metric.aggregate().item())
    results["hd95"] = float(hd95_metric.aggregate().item())

    dice_metric.reset()
    hd95_metric.reset()

    return results


def plot_history(history, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    val_epochs = history["val_epoch"]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    if len(val_epochs) == len(history["val_dice"]):
        plt.plot(val_epochs, history["val_dice"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_dice_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    if len(val_epochs) == len(history["val_hd95"]):
        plt.plot(val_epochs, history["val_hd95"], label="val_hd95")
    plt.xlabel("Epoch")
    plt.ylabel("HD95")
    plt.title("Validation HD95")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_hd95_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["lr"], label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lr_curve.png", dpi=150)
    plt.close()


def train_fold(args, fold):
    output_dir = Path(args.output_dir)
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(output_dir / f"fold_{fold}_train.csv")
    val_df = pd.read_csv(output_dir / f"fold_{fold}_val.csv")

    log.info(f"Fold {fold}: train={len(train_df)} | val={len(val_df)}")

    train_cases = make_case_dicts(train_df)
    val_cases = make_case_dicts(val_df)

    train_transforms, val_transforms = get_transforms(
        patch_size=tuple(args.patch_size),
        spacing=tuple(args.spacing),
        num_samples=args.num_samples,
    )

    train_ds = CacheDataset(
        data=train_cases,
        transform=train_transforms,
        cache_rate=args.cache_rate,
        num_workers=NUM_WORKERS,
    )
    val_ds = Dataset(data=val_cases, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model().to(DEVICE)

    criterion = DiceFocalLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(fold_dir / "tensorboard"))

    history = {
        "train_loss": [],
        "val_epoch": [],
        "val_dice": [],
        "val_hd95": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_specificity": [],
        "val_accuracy": [],
        "val_f1": [],
        "lr": [],
    }

    best_dice = -1.0
    patience_counter = 0
    best_path = fold_dir / "best_model.pt"

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{args.epochs}", leave=False)

        for batch in pbar:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=images.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        scheduler.step()

        history["train_loss"].append(float(loss_meter.avg))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if writer:
            writer.add_scalar("train/loss", loss_meter.avg, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs:
            metrics = validate(model, val_loader, roi_size=tuple(args.patch_size))

            history["val_epoch"].append(epoch + 1)
            history["val_dice"].append(metrics["dice"])
            history["val_hd95"].append(metrics["hd95"])
            history["val_iou"].append(metrics["iou"])
            history["val_precision"].append(metrics["precision"])
            history["val_recall"].append(metrics["recall"])
            history["val_specificity"].append(metrics["specificity"])
            history["val_accuracy"].append(metrics["accuracy"])
            history["val_f1"].append(metrics["f1"])

            log.info(
                f"Fold {fold} Epoch {epoch+1}: "
                f"Dice={metrics['dice']:.4f} "
                f"IoU={metrics['iou']:.4f} "
                f"HD95={metrics['hd95']:.4f} "
                f"Prec={metrics['precision']:.4f} "
                f"Rec={metrics['recall']:.4f}"
            )

            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch + 1)

            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "best_dice": best_dice,
                        "config": {
                            "patch_size": list(args.patch_size),
                            "spacing": list(args.spacing),
                            "num_samples": args.num_samples,
                        },
                    },
                    best_path,
                )

                with open(fold_dir / "best_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                log.info(f"Saved best model for fold {fold} (dice={best_dice:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log.info(f"Early stopping fold {fold} at epoch {epoch+1}")
                    break

        max_len = max(len(v) for v in history.values())
        padded = {}
        for k, vals in history.items():
            vals = list(vals)
            if len(vals) < max_len:
                vals = vals + [None] * (max_len - len(vals))
            padded[k] = vals

        pd.DataFrame(padded).to_csv(fold_dir / "training_history.csv", index=False)

        with open(fold_dir / "config.json", "w") as f:
            json.dump(
                {
                    "device": str(DEVICE),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "val_every": args.val_every,
                    "patience": args.patience,
                    "patch_size": list(args.patch_size),
                    "spacing": list(args.spacing),
                    "num_samples": args.num_samples,
                    "cache_rate": args.cache_rate,
                },
                f,
                indent=2,
            )

    plot_history(history, fold_dir / "plots")

    if writer:
        writer.close()


def main(args):
    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    folds = range(5) if args.fold == -1 else [args.fold]

    for fold in folds:
        log.info("=" * 70)
        log.info(f"TRAINING FOLD {fold}")
        log.info("=" * 70)
        train_fold(args, fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Folder containing fold CSVs from prepare_resunet_dataset.py")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--fold", type=int, default=-1, help="-1 means all folds")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--spacing", nargs=3, type=float, default=[1.5, 1.5, 3.0])
    parser.add_argument("--num_samples", type=int, default=4, help="patches sampled per volume in training")
    parser.add_argument("--cache_rate", type=float, default=0.5)

    args = parser.parse_args()
    main(args)