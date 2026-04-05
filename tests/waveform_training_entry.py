"""Minimal training entry that bridges waveform data to the spectral model."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.training_step import SpeechEnhancementTrainingStep, TrainingStepOutput
from utils.batch import WaveformBatch
from utils.complex_utils import complex_power_decompress
from utils.dataset import build_a4bs250h_dataloader
from utils.metrics import batch_waveform_metrics
from utils.waveform_adapter import WaveformToSpectralBatchAdapter


@dataclass(frozen=True)
class WaveformTrainingEntryConfig:
    """Configuration for the minimal waveform-domain training entry."""

    dataset_root: str
    split: str = "train"
    batch_size: int = 1
    shuffle: bool | None = None
    num_workers: int = 0
    drop_last: bool = False
    learning_rate: float = 1e-3
    gpus: str = "0"  # Comma-separated GPU IDs
    use_amp: bool = False
    amp_dtype: str = "float16"
    persistent_workers: bool | None = None


@dataclass(frozen=True)
class TrainingIterationSummary:
    """
    Summary returned by one waveform training / evaluation iteration.

    Shape:
        model_output.enhanced_complex: [B, 2, T, 161]
        total_loss / ri_loss / mag_loss: scalar tensors
        sisnr/pesq/estoi: float metrics
    """

    step_output: TrainingStepOutput
    batch_size: int
    did_optimizer_step: bool
    sisnr: float
    pesq: float
    estoi: float


@dataclass(frozen=True)
class EpochSummary:
    """
    Aggregated metrics for one epoch-level pass over a dataloader.

    Shape:
        loss / ri_loss / mag_loss / sisnr / pesq / estoi: python floats averaged
        over all examples.
    """

    split: str
    training: bool
    num_batches: int
    num_examples: int
    loss: float
    ri_loss: float
    mag_loss: float
    sisnr: float
    pesq: float
    estoi: float


class WaveformTrainingEntry(nn.Module):
    """
    Minimal runnable training entry for waveform-domain dataset batches.

    This entry wraps exactly the current mainline:
        1. DataLoader emits waveform tensors `[B, 64000]`
        2. `WaveformToSpectralBatchAdapter` converts them to `[B, 2, T, 161]`
        3. `SpeechEnhancementTrainingStep` runs forward + loss
        4. If `training=True`, optimizer step is applied

    engineering assumption:
        The paper does not define an application-level training entry object.
        This class is a thin project runtime utility around the existing model,
        adapter, and loss modules.
    """

    def __init__(
        self,
        cfg: WaveformTrainingEntryConfig,
        adapter: WaveformToSpectralBatchAdapter | None = None,
        training_step: SpeechEnhancementTrainingStep | None = None,
        optimizer: Optimizer | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.adapter = adapter or WaveformToSpectralBatchAdapter()
        self.training_step = training_step or SpeechEnhancementTrainingStep()
        self.gpus = [int(g) for g in cfg.gpus.split(',')]
        self.device = torch.device(f"cuda:{self.gpus[0]}") if self.gpus else torch.device("cpu")
        self.optimizer = optimizer or Adam(self.training_step.parameters(), lr=self.cfg.learning_rate)
        self.use_amp = self.cfg.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if self.cfg.amp_dtype == "float16" else torch.bfloat16
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Multi-GPU support
        if len(self.gpus) > 1:
            self.training_step.to(self.device)
            self.training_step.model = nn.DataParallel(self.training_step.model, device_ids=self.gpus)
            self.adapter = self.adapter.to(self.device)
        else:
            self.adapter.to(self.device)
            self.training_step.to(self.device)

    def build_dataloader(self, split: str | None = None, shuffle: bool | None = None) -> DataLoader:
        """Build a waveform DataLoader using the existing dataset interface."""
        return build_a4bs250h_dataloader(
            root=self.cfg.dataset_root,
            split=split or self.cfg.split,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle if shuffle is None else shuffle,
            num_workers=self.cfg.num_workers,
            drop_last=self.cfg.drop_last,
            pin_memory=True,
            persistent_workers=self.cfg.persistent_workers,
        )

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def run_batch(
        self,
        waveform_batch: WaveformBatch | dict[str, torch.Tensor],
        training: bool = True,
        compute_metrics: bool = True,
        metrics_max_examples: int | None = None,
        metrics_include_pesq_estoi: bool = True,
    ) -> TrainingIterationSummary:
        """Run one collated waveform batch through adapter, model, loss, and optional optimizer step."""
        # Move waveform batch to device for GPU STFT
        if isinstance(waveform_batch, dict):
            waveform_batch = {k: v.to(self.device) for k, v in waveform_batch.items()}
        else:
            waveform_batch = waveform_batch.to(self.device)

        spectral_batch = self.adapter(waveform_batch)
        spectral_batch = spectral_batch.to(device=self.device)

        self.training_step.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with self._autocast_context():
                step_output = self.training_step(spectral_batch)
        if training:
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(step_output.total_loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                step_output.total_loss.backward()
                self.optimizer.step()

        sisnr = pesq = estoi = float('nan')
        if compute_metrics:
            clean_complex_for_istft = complex_power_decompress(
                spectral_batch.clean_ac,
                exponent=self.adapter.compression_exponent,
            )
            enhanced_complex_for_istft = complex_power_decompress(
                step_output.model_output.enhanced_complex,
                exponent=self.adapter.compression_exponent,
            )
            # waveform metrics: use clean clean waveform and enhanced reconstructed waveform.
            clean_waveform = self.adapter.stft_processor.complex_to_waveform(
                clean_complex_for_istft,
                length=waveform_batch["clean"].shape[-1] if isinstance(waveform_batch, dict) else waveform_batch.clean.shape[-1],
            )
            enhanced_waveform = self.adapter.stft_processor.complex_to_waveform(
                enhanced_complex_for_istft,
                length=clean_waveform.shape[-1],
            )
            metrics = batch_waveform_metrics(
                clean_waveform,
                enhanced_waveform,
                sample_rate=self.adapter.stft_cfg.sample_rate,
                max_examples=metrics_max_examples,
                compute_pesq_estoi=metrics_include_pesq_estoi,
            )
            sisnr = metrics["sisnr"]
            pesq = metrics["pesq"]
            estoi = metrics["estoi"]

        return TrainingIterationSummary(
            step_output=step_output,
            batch_size=spectral_batch.batch_size,
            did_optimizer_step=training,
            sisnr=sisnr,
            pesq=pesq,
            estoi=estoi,
        )

    @torch.no_grad()
    def preview_audio_triplet(
        self,
        waveform_batch: WaveformBatch | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build one-sample waveform preview for TensorBoard audio logging."""
        if isinstance(waveform_batch, dict):
            waveform_batch = {k: v.to(self.device) for k, v in waveform_batch.items()}
            clean_source = waveform_batch["clean"]
            mixture_source = waveform_batch["mixture"]
        else:
            waveform_batch = waveform_batch.to(self.device)
            clean_source = waveform_batch.clean
            mixture_source = waveform_batch.mixture

        self.training_step.train(False)

        spectral_batch = self.adapter(waveform_batch)
        spectral_batch = spectral_batch.to(device=self.device)
        with self._autocast_context():
            step_output = self.training_step(spectral_batch)

        enhanced_complex_for_istft = complex_power_decompress(
            step_output.model_output.enhanced_complex,
            exponent=self.adapter.compression_exponent,
        )
        enhanced_waveform = self.adapter.stft_processor.complex_to_waveform(
            enhanced_complex_for_istft,
            length=clean_source.shape[-1],
        )

        return {
            "enhanced": enhanced_waveform[0].detach().float().cpu(),
            "mixture": mixture_source[0].detach().float().cpu(),
            "clean": clean_source[0].detach().float().cpu(),
        }

    @torch.no_grad()
    def fixed_batch_prediction_stats(
        self,
        waveform_batch: WaveformBatch | dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Return one-sample prediction/target summary stats on a fixed batch."""
        if isinstance(waveform_batch, dict):
            waveform_batch = {k: v.to(self.device) for k, v in waveform_batch.items()}
        else:
            waveform_batch = waveform_batch.to(self.device)

        was_training = self.training_step.training
        self.training_step.train(False)
        spectral_batch = self.adapter(waveform_batch)
        spectral_batch = spectral_batch.to(device=self.device)
        with self._autocast_context():
            step_output = self.training_step(spectral_batch)
        self.training_step.train(was_training)

        pred = step_output.model_output.enhanced_complex[0].detach().float()
        target = spectral_batch.clean_ac[0].detach().float()
        return {
            "pred_mean": float(pred.mean().item()),
            "pred_std": float(pred.std(unbiased=False).item()),
            "target_mean": float(target.mean().item()),
            "target_std": float(target.std(unbiased=False).item()),
        }

    def run_loader(
        self,
        dataloader: DataLoader,
        max_batches: int | None = None,
        training: bool = True,
        compute_metrics: bool = False,
        metrics_max_examples: int | None = None,
        metrics_include_pesq_estoi: bool = True,
        writer: SummaryWriter | None = None,
        global_step: int = 0,
        log_every_steps: int = 0,
        debug_print_first_steps: int = 0,
        fixed_probe_batch: WaveformBatch | dict[str, torch.Tensor] | None = None,
        fixed_probe_every_steps: int = 0,
        fixed_probe_first_steps: int = 0,
        return_details: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], list[dict[str, float]], int]:
        """Run a dataloader for a small training / evaluation sweep and return averaged losses and per-batch metrics."""
        import time
        total_loss = 0.0
        total_ri = 0.0
        total_mag = 0.0
        total_sisnr = 0.0
        total_pesq = 0.0
        total_estoi = 0.0
        total_examples = 0
        num_batches = 0
        batch_list = []

        for batch_idx, waveform_batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            start_time = time.time()
            summary = self.run_batch(
                waveform_batch,
                training=training,
                compute_metrics=compute_metrics,
                metrics_max_examples=metrics_max_examples,
                metrics_include_pesq_estoi=metrics_include_pesq_estoi,
            )
            iter_time = time.time() - start_time

            # For now, approximate data_time as 0 and compute_time as iter_time
            # TODO: More precise timing would require separating dataloader iteration from device transfer
            data_time = 0.0
            compute_time = iter_time

            batch_size = summary.batch_size
            total_examples += batch_size
            num_batches += 1
            total_loss += float(summary.step_output.total_loss.detach()) * batch_size
            total_ri += float(summary.step_output.ri_loss.detach()) * batch_size
            total_mag += float(summary.step_output.mag_loss.detach()) * batch_size
            if compute_metrics:
                total_sisnr += float(summary.sisnr) * batch_size
                total_pesq += float(summary.pesq) * batch_size
                total_estoi += float(summary.estoi) * batch_size

            batch_metrics = {
                "loss": float(summary.step_output.total_loss.detach()),
                "ri_loss": float(summary.step_output.ri_loss.detach()),
                "mag_loss": float(summary.step_output.mag_loss.detach()),
                "iter_time": iter_time,
                "data_time": data_time,
                "compute_time": compute_time
            }
            batch_list.append(batch_metrics)

            # Real-time step logging for TensorBoard (training only)
            if training and writer is not None:
                writer.add_scalar("Loss/train_step", batch_metrics["loss"], global_step)
                writer.add_scalar("Loss/ri_step", batch_metrics["ri_loss"], global_step)
                writer.add_scalar("Loss/mag_step", batch_metrics["mag_loss"], global_step)
                writer.add_scalar("Time/iter", batch_metrics["iter_time"], global_step)
                writer.add_scalar("Time/data", batch_metrics["data_time"], global_step)
                writer.add_scalar("Time/compute", batch_metrics["compute_time"], global_step)
                writer.add_scalar("Learning_Rate", float(self.optimizer.param_groups[0]["lr"]), global_step)
                writer.flush()

                if log_every_steps > 0 and global_step % log_every_steps == 0:
                    print(
                        f"step {global_step}: loss={batch_metrics['loss']:.6f} ri={batch_metrics['ri_loss']:.6f} "
                        f"mag={batch_metrics['mag_loss']:.6f} iter_time={batch_metrics['iter_time']:.3f}s"
                    )

            if training and debug_print_first_steps > 0 and global_step < debug_print_first_steps:
                print(
                    f"debug_step {global_step}: loss={batch_metrics['loss']:.6f} "
                    f"ri={batch_metrics['ri_loss']:.6f} mag={batch_metrics['mag_loss']:.6f}"
                )

            should_probe = (
                training
                and fixed_probe_batch is not None
                and fixed_probe_every_steps > 0
                and (fixed_probe_first_steps <= 0 or global_step < fixed_probe_first_steps)
                and (global_step % fixed_probe_every_steps == 0)
            )
            if should_probe:
                stats = self.fixed_batch_prediction_stats(fixed_probe_batch)
                print(
                    "fixed_probe "
                    f"step={global_step} "
                    f"pred_mean={stats['pred_mean']:.6f} pred_std={stats['pred_std']:.6f} "
                    f"target_mean={stats['target_mean']:.6f} target_std={stats['target_std']:.6f}"
                )

            global_step += 1

        if total_examples == 0:
            raise ValueError("run_loader received no batches to process.")

        result = {
            "num_batches": float(num_batches),
            "num_examples": float(total_examples),
            "loss": total_loss / total_examples,
            "ri_loss": total_ri / total_examples,
            "mag_loss": total_mag / total_examples,
        }
        if compute_metrics:
            result.update({
                "sisnr": total_sisnr / total_examples,
                "pesq": total_pesq / total_examples,
                "estoi": total_estoi / total_examples,
            })
        if return_details:
            return result, batch_list, global_step
        return result

    @staticmethod
    def _epoch_summary_from_metrics(
        metrics: dict[str, float],
        *,
        split: str,
        training: bool,
    ) -> EpochSummary:
        """Convert averaged loader metrics into a typed epoch summary."""
        return EpochSummary(
            split=split,
            training=training,
            num_batches=int(metrics["num_batches"]),
            num_examples=int(metrics["num_examples"]),
            loss=float(metrics["loss"]),
            ri_loss=float(metrics["ri_loss"]),
            mag_loss=float(metrics["mag_loss"]),
            sisnr=float(metrics.get("sisnr", float("nan"))),
            pesq=float(metrics.get("pesq", float("nan"))),
            estoi=float(metrics.get("estoi", float("nan"))),
        )

    def train_one_epoch(
        self,
        dataloader: DataLoader | None = None,
        *,
        max_batches: int | None = None,
        split: str | None = None,
        compute_metrics: bool = False,
        writer: SummaryWriter | None = None,
        global_step: int = 0,
        log_every_steps: int = 0,
        debug_print_first_steps: int = 0,
        fixed_probe_batch: WaveformBatch | dict[str, torch.Tensor] | None = None,
        fixed_probe_every_steps: int = 0,
        fixed_probe_first_steps: int = 0,
        metrics_max_examples: int | None = None,
        metrics_include_pesq_estoi: bool = True,
        return_details: bool = False,
    ) -> EpochSummary | tuple[EpochSummary, list[dict[str, float]], int]:
        """
        Run one training epoch over waveform batches.

        Args:
            dataloader: Optional external waveform dataloader. If omitted, one is
                built from the current config.
            max_batches: Optional limit for smoke tests or short runs.
            split: Optional dataset split override used only when building the
                dataloader internally.
        """
        resolved_split = split or self.cfg.split
        loader = dataloader or self.build_dataloader(split=resolved_split, shuffle=True)
        metrics, batch_list, global_step = self.run_loader(
            loader,
            max_batches=max_batches,
            training=True,
            compute_metrics=compute_metrics,
            metrics_max_examples=metrics_max_examples,
            metrics_include_pesq_estoi=metrics_include_pesq_estoi,
            writer=writer,
            global_step=global_step,
            log_every_steps=log_every_steps,
            debug_print_first_steps=debug_print_first_steps,
            fixed_probe_batch=fixed_probe_batch,
            fixed_probe_every_steps=fixed_probe_every_steps,
            fixed_probe_first_steps=fixed_probe_first_steps,
            return_details=True,
        )
        summary = self._epoch_summary_from_metrics(metrics, split=resolved_split, training=True)
        if return_details:
            return summary, batch_list, global_step
        return summary

    def validate_one_epoch(
        self,
        dataloader: DataLoader | None = None,
        *,
        max_batches: int | None = None,
        split: str = "valid",
        compute_metrics: bool = True,
        metrics_max_examples: int | None = None,
        metrics_include_pesq_estoi: bool = True,
        return_details: bool = False,
    ) -> EpochSummary | tuple[EpochSummary, list[dict[str, float]]]:
        """
        Run one validation epoch over waveform batches without optimizer steps.

        Args:
            dataloader: Optional external waveform dataloader. If omitted, one is
                built from the requested validation split.
            max_batches: Optional limit for smoke tests or short runs.
            split: Validation split used when building the dataloader internally.
        """
        loader = dataloader or self.build_dataloader(split=split, shuffle=False)
        metrics, batch_list, _ = self.run_loader(
            loader,
            max_batches=max_batches,
            training=False,
            compute_metrics=compute_metrics,
            metrics_max_examples=metrics_max_examples,
            metrics_include_pesq_estoi=metrics_include_pesq_estoi,
            return_details=True,
        )
        summary = self._epoch_summary_from_metrics(metrics, split=split, training=False)
        if return_details:
            return summary, batch_list
        return summary
