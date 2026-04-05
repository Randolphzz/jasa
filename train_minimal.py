"""Minimal executable training script for waveform-domain training."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.waveform_training_entry import EpochSummary, WaveformTrainingEntry, WaveformTrainingEntryConfig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal train/valid loop for the bone-air speech enhancement model."
    )
    parser.add_argument("--dataset-root", type=str, default="/mnt/kkl/A4BS_250h", help="Root directory of A4BS_250h.")
    parser.add_argument("--train-split", type=str, default="train", choices=("train", "valid", "test"))
    parser.add_argument("--valid-split", type=str, default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers (recommended: 4x num GPUs)")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated list of GPU IDs to use, e.g., '0,1'. Set num-workers to 4x num-GPUs.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-max-batches", type=int, default=None)
    parser.add_argument("--valid-max-batches", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="artifacts/minimal_train")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--lr-step-size", type=int, default=0, help="Enable StepLR when > 0.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Decay factor for StepLR.")
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also save an epoch checkpoint alongside the best checkpoint.",
    )
    parser.add_argument("--log-every-steps", type=int, default=50, help="Log metrics every N training steps.")
    parser.add_argument("--random-data", action="store_true", help="Use random waveform data for speed/interface test.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for isolated logging.")
    parser.add_argument("--use-amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--valid-metrics-every-epochs", type=int, default=5, help="Run PESQ/eSTOI every N epochs.")
    parser.add_argument("--valid-metrics-max-batches", type=int, default=4, help="Validation batch limit for heavy metrics.")
    parser.add_argument("--valid-metrics-max-examples", type=int, default=8, help="Waveform example cap for heavy metrics.")
    parser.add_argument(
        "--debug-print-first-steps",
        type=int,
        default=0,
        help="If > 0, print every training batch loss for the first N global steps.",
    )
    parser.add_argument(
        "--fixed-probe-every-steps",
        type=int,
        default=0,
        help="If > 0, run fixed-batch pred/target mean/std probe every N global steps.",
    )
    parser.add_argument(
        "--fixed-probe-first-steps",
        type=int,
        default=300,
        help="Only run fixed-batch probe within the first N global steps (<=0 means always).",
    )
    return parser.parse_args(argv)


class RandomWaveformDataset(Dataset):
    """Generate random waveform data for speed/pipe sanity checks."""

    def __init__(self, length: int = 64000, batch_count: int = 12000):
        self.length = length
        self.batch_count = batch_count

    def __len__(self) -> int:
        return self.batch_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        waveform = torch.randn(self.length, dtype=torch.float32)
        return {
            "mixture": waveform,
            "bc": waveform,
            "clean": waveform,
            "bc_channel": 1,
        }


def build_random_dataloader(batch_size: int, num_workers: int, split: str = "train") -> DataLoader:
    return DataLoader(
        RandomWaveformDataset(),
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
        collate_fn=lambda batch: {
            "mixture": torch.stack([s["mixture"] for s in batch], dim=0),
            "bc": torch.stack([s["bc"] for s in batch], dim=0),
            "clean": torch.stack([s["clean"] for s in batch], dim=0),
            "bc_channel": torch.tensor([s["bc_channel"] for s in batch], dtype=torch.long),
        },
    )


def _summary_to_dict(summary: EpochSummary, epoch: int) -> dict[str, float | int | bool | str]:
    return {
        "epoch": epoch,
        "split": summary.split,
        "training": summary.training,
        "num_batches": summary.num_batches,
        "num_examples": summary.num_examples,
        "loss": summary.loss,
        "ri_loss": summary.ri_loss,
        "mag_loss": summary.mag_loss,
        "sisnr": summary.sisnr,
        "pesq": summary.pesq,
        "estoi": summary.estoi,
    }


def _format_summary(summary: EpochSummary) -> str:
    mode = "train" if summary.training else "valid"
    return (
        f"{mode}: loss={summary.loss:.6f} "
        f"ri={summary.ri_loss:.6f} "
        f"mag={summary.mag_loss:.6f} "
        f"sisnr={summary.sisnr:.3f} "
        f"pesq={summary.pesq:.3f} "
        f"estoi={summary.estoi:.3f} "
        f"batches={summary.num_batches} "
        f"examples={summary.num_examples}"
    )


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    best_valid_loss: float,
    global_step: int,
    train_summary: EpochSummary,
    valid_summary: EpochSummary,
    entry: WaveformTrainingEntry,
    scheduler: StepLR | None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_valid_loss": best_valid_loss,
        "config": vars(entry.cfg),
        "train_summary": _summary_to_dict(train_summary, epoch),
        "valid_summary": _summary_to_dict(valid_summary, epoch),
        "model_state_dict": entry.training_step.state_dict(),
        "optimizer_state_dict": entry.optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def _save_checkpoint_pair(
    run_dir: Path,
    base_save_dir: Path,
    relative_name: str,
    *,
    epoch: int,
    best_valid_loss: float,
    global_step: int,
    train_summary: EpochSummary,
    valid_summary: EpochSummary,
    entry: WaveformTrainingEntry,
    scheduler: StepLR | None,
) -> None:
    _save_checkpoint(
        run_dir / relative_name,
        epoch=epoch,
        best_valid_loss=best_valid_loss,
        global_step=global_step,
        train_summary=train_summary,
        valid_summary=valid_summary,
        entry=entry,
        scheduler=scheduler,
    )
    if run_dir != base_save_dir:
        _save_checkpoint(
            base_save_dir / relative_name,
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            global_step=global_step,
            train_summary=train_summary,
            valid_summary=valid_summary,
            entry=entry,
            scheduler=scheduler,
        )


def _build_scheduler(args: argparse.Namespace, entry: WaveformTrainingEntry) -> StepLR | None:
    if args.lr_step_size <= 0:
        return None
    return StepLR(entry.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    entry: WaveformTrainingEntry,
    scheduler: StepLR | None,
) -> tuple[int, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location=entry.device)
    entry.training_step.load_state_dict(checkpoint["model_state_dict"])
    entry.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
    global_step = int(checkpoint.get("global_step", 0))
    return start_epoch, best_valid_loss, global_step


def run_training(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_save_dir = Path(args.save_dir)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Create isolated run directory
    if args.resume:
        # Resume from existing run (resume path should be run_xxx/last.pt or run_xxx/best.pt)
        resume_path = Path(args.resume)
        run_dir = resume_path.parent  # artifacts/minimal_train/run_xxx
    else:
        # Create new run
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"run_{timestamp}"
        run_dir = base_save_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    root_metrics_path = base_save_dir / "metrics.jsonl"
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    entry = WaveformTrainingEntry(
        WaveformTrainingEntryConfig(
            dataset_root=args.dataset_root,
            split=args.train_split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            gpus=args.gpus,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
        )
    )
    scheduler = _build_scheduler(args, entry)

    best_valid_loss = float("inf")
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        start_epoch, best_valid_loss, global_step = _load_checkpoint(
            Path(args.resume),
            entry=entry,
            scheduler=scheduler,
        )

    if args.random_data:
        train_loader = build_random_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, split=args.train_split)
        valid_loader = build_random_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, split=args.valid_split)
    else:
        train_loader = entry.build_dataloader(split=args.train_split, shuffle=True)
        valid_loader = entry.build_dataloader(split=args.valid_split, shuffle=False)

    fixed_probe_batch = None
    if args.fixed_probe_every_steps > 0:
        try:
            fixed_probe_batch = next(iter(train_loader))
        except StopIteration:
            fixed_probe_batch = None

    metrics_mode = "a" if args.resume is not None and root_metrics_path.exists() else "w"

    with ExitStack() as stack:
        metrics_file = stack.enter_context(metrics_path.open(metrics_mode, encoding="utf-8"))
        if root_metrics_path == metrics_path:
            root_metrics_file = metrics_file
        else:
            root_metrics_file = stack.enter_context(root_metrics_path.open(metrics_mode, encoding="utf-8"))
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()

            # Training
            train_start_time = time.time()
            train_summary, train_batch_list, global_step = entry.train_one_epoch(
                train_loader,
                max_batches=args.train_max_batches,
                split=args.train_split,
                compute_metrics=False,
                writer=writer,
                global_step=global_step,
                log_every_steps=args.log_every_steps,
                debug_print_first_steps=args.debug_print_first_steps,
                fixed_probe_batch=fixed_probe_batch,
                fixed_probe_every_steps=args.fixed_probe_every_steps,
                fixed_probe_first_steps=args.fixed_probe_first_steps,
                return_details=True,
            )
            train_time = time.time() - train_start_time

            valid_loss_batches = args.valid_max_batches
            if valid_loss_batches is None:
                valid_loss_batches = max(1, len(valid_loader) // 4)

            valid_loss_start_time = time.time()
            valid_summary, valid_batch_list = entry.validate_one_epoch(
                valid_loader,
                max_batches=valid_loss_batches,
                split=args.valid_split,
                compute_metrics=False,
                return_details=True,
            )
            valid_loss_time = time.time() - valid_loss_start_time

            should_run_heavy_metrics = args.valid_metrics_every_epochs > 0 and epoch % args.valid_metrics_every_epochs == 0
            valid_metrics_time = 0.0
            if should_run_heavy_metrics:
                valid_metrics_start_time = time.time()
                metrics_summary, _ = entry.validate_one_epoch(
                    valid_loader,
                    max_batches=min(args.valid_metrics_max_batches, len(valid_loader)),
                    split=args.valid_split,
                    compute_metrics=True,
                    metrics_max_examples=args.valid_metrics_max_examples,
                    metrics_include_pesq_estoi=True,
                    return_details=True,
                )
                valid_metrics_time = time.time() - valid_metrics_start_time
                valid_summary = EpochSummary(
                    split=valid_summary.split,
                    training=valid_summary.training,
                    num_batches=valid_summary.num_batches,
                    num_examples=valid_summary.num_examples,
                    loss=valid_summary.loss,
                    ri_loss=valid_summary.ri_loss,
                    mag_loss=valid_summary.mag_loss,
                    sisnr=metrics_summary.sisnr,
                    pesq=metrics_summary.pesq,
                    estoi=metrics_summary.estoi,
                )

            epoch_time = time.time() - epoch_start_time

            # Epoch-level logging
            train_record = _summary_to_dict(train_summary, epoch)
            train_record["lr"] = float(entry.optimizer.param_groups[0]["lr"])
            valid_record = _summary_to_dict(valid_summary, epoch)
            valid_record["lr"] = float(entry.optimizer.param_groups[0]["lr"])
            valid_record["train_time_sec"] = train_time
            valid_record["valid_loss_time_sec"] = valid_loss_time
            valid_record["valid_metrics_time_sec"] = valid_metrics_time
            valid_record["epoch_time_sec"] = epoch_time

            metrics_file.write(json.dumps(train_record, ensure_ascii=True) + "\n")
            metrics_file.write(json.dumps(valid_record, ensure_ascii=True) + "\n")
            metrics_file.flush()
            if root_metrics_file is not metrics_file:
                root_metrics_file.write(json.dumps(train_record, ensure_ascii=True) + "\n")
                root_metrics_file.write(json.dumps(valid_record, ensure_ascii=True) + "\n")
                root_metrics_file.flush()

            # TensorBoard epoch-level
            writer.add_scalar("Loss/Train", train_summary.loss, epoch)
            writer.add_scalar("Loss/Valid", valid_summary.loss, epoch)
            writer.add_scalar("Loss/RI_Train", train_summary.ri_loss, epoch)
            writer.add_scalar("Loss/RI_Valid", valid_summary.ri_loss, epoch)
            writer.add_scalar("Loss/Mag_Train", train_summary.mag_loss, epoch)
            writer.add_scalar("Loss/Mag_Valid", valid_summary.mag_loss, epoch)
            writer.add_scalar("SISNR/Valid", valid_summary.sisnr, epoch)
            writer.add_scalar("PESQ/Valid", valid_summary.pesq, epoch)
            writer.add_scalar("ESTOI/Valid", valid_summary.estoi, epoch)
            writer.add_scalar("Learning_Rate", float(entry.optimizer.param_groups[0]["lr"]), epoch)

            # Epoch-level audio preview (one sample) for listening checks.
            try:
                preview_batch = next(iter(valid_loader))
                audio_triplet = entry.preview_audio_triplet(preview_batch)
                sample_rate = entry.adapter.stft_cfg.sample_rate
                writer.add_audio("Audio/valid_enhanced", audio_triplet["enhanced"], epoch, sample_rate=sample_rate)
                writer.add_audio("Audio/valid_mixture", audio_triplet["mixture"], epoch, sample_rate=sample_rate)
                writer.add_audio("Audio/valid_clean", audio_triplet["clean"], epoch, sample_rate=sample_rate)
            except Exception as exc:
                print(f"warning: failed to log audio preview at epoch {epoch}: {exc}")

            print(f"epoch {epoch}/{args.epochs}")
            print(_format_summary(valid_summary))  # Only valid since train is nan
            print(f"lr={float(entry.optimizer.param_groups[0]['lr']):.6e}")
            print(
                f"time: train={train_time:.1f}s valid_loss={valid_loss_time:.1f}s "
                f"valid_metrics={valid_metrics_time:.1f}s total={epoch_time:.1f}s"
            )

            if scheduler is not None:
                scheduler.step()

            if args.save_every_epoch:
                _save_checkpoint_pair(
                    run_dir,
                    base_save_dir,
                    f"epoch_{epoch}.pt",
                    epoch=epoch,
                    best_valid_loss=best_valid_loss,
                    global_step=global_step,
                    train_summary=train_summary,
                    valid_summary=valid_summary,
                    entry=entry,
                    scheduler=scheduler,
                )

            if valid_summary.loss < best_valid_loss:
                best_valid_loss = valid_summary.loss
                _save_checkpoint_pair(
                    run_dir,
                    base_save_dir,
                    "best.pt",
                    epoch=epoch,
                    best_valid_loss=best_valid_loss,
                    global_step=global_step,
                    train_summary=train_summary,
                    valid_summary=valid_summary,
                    entry=entry,
                    scheduler=scheduler,
                )

            _save_checkpoint_pair(
                run_dir,
                base_save_dir,
                "last.pt",
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                global_step=global_step,
                train_summary=train_summary,
                valid_summary=valid_summary,
                entry=entry,
                scheduler=scheduler,
            )

    writer.close()
    print(f"artifacts saved to {run_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_training(argv)


if __name__ == "__main__":
    raise SystemExit(main())
