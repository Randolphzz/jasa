"""Inference script for dual-modal AC/BC speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import soundfile as sf
import torch

from models.training_step import SpeechEnhancementTrainingStep
from utils.complex_utils import complex_power_decompress
from utils.waveform_adapter import WaveformToSpectralBatchAdapter


DEFAULT_CHECKPOINT = "artifacts/minimal_train/run_20260331_214918/best.pt"
DEFAULT_MIX_WAV = "/mnt/kkl/A4BS_250h/valid/0/mix.wav"
DEFAULT_OUTPUT_WAV = "test_wav/infer_valid0.wav"
DEFAULT_SAVE_MIX_WAV = "test_wav/mix.wav"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline inference from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint (.pt).")
    parser.add_argument("--mix-wav", type=str, default=DEFAULT_MIX_WAV, help="Input mix.wav path (multi-channel).")
    parser.add_argument("--output-wav", type=str, default=DEFAULT_OUTPUT_WAV, help="Enhanced waveform output path.")
    parser.add_argument("--ac-channel", type=int, default=0, help="AC/noisy channel index in mix.wav.")
    parser.add_argument("--bc-channel", type=int, default=1, help="BC channel index in mix.wav.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate.")
    parser.add_argument("--compression-exponent", type=float, default=0.5, help="Spectral compression exponent.")
    parser.add_argument("--device", type=str, default=None, help="Device such as cuda:0 or cpu.")
    parser.add_argument("--use-amp", action="store_true", help="Enable autocast when running on CUDA.")
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=("float16", "bfloat16"))
    parser.add_argument(
        "--save-mix-wav",
        type=str,
        default=DEFAULT_SAVE_MIX_WAV,
        help="Path to save a copy of the source multi-channel mix wav.",
    )
    parser.add_argument("--save-input-ac", type=str, default=None, help="Optional path to save AC channel waveform.")
    parser.add_argument("--save-input-bc", type=str, default=None, help="Optional path to save BC channel waveform.")
    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict is not a dictionary.")
    return state


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model.module."):
            new_key = "model." + new_key[len("model.module.") :]
        normalized[new_key] = value
    return normalized


def _load_model_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> SpeechEnhancementTrainingStep:
    training_step = SpeechEnhancementTrainingStep().to(device)
    state_dict = _load_checkpoint_state(checkpoint_path, device)

    candidates = [state_dict, _normalize_state_dict_keys(state_dict)]
    loaded = False
    error_messages: list[str] = []
    for candidate in candidates:
        try:
            training_step.load_state_dict(candidate, strict=True)
            loaded = True
            break
        except RuntimeError as exc:
            error_messages.append(str(exc))

    if not loaded:
        raise RuntimeError("Failed to load checkpoint into model.\n" + "\n---\n".join(error_messages))

    training_step.eval()
    return training_step


def _read_mix_channels(
    mix_wav_path: Path,
    *,
    ac_channel: int,
    bc_channel: int,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    mix, sr = sf.read(str(mix_wav_path), dtype="float32")
    if sr != sample_rate:
        raise ValueError(f"Unexpected sample rate: {sr}, expected {sample_rate}.")
    if mix.ndim != 2:
        raise ValueError(f"mix wav must be multi-channel with shape [T, C], got {mix.shape}.")
    num_channels = mix.shape[1]
    if ac_channel < 0 or ac_channel >= num_channels:
        raise ValueError(f"ac-channel {ac_channel} out of range for {num_channels} channels.")
    if bc_channel < 0 or bc_channel >= num_channels:
        raise ValueError(f"bc-channel {bc_channel} out of range for {num_channels} channels.")

    ac = torch.from_numpy(mix[:, ac_channel]).float().unsqueeze(0)
    bc = torch.from_numpy(mix[:, bc_channel]).float().unsqueeze(0)
    return ac, bc, sr


def main() -> int:
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint)
    mix_wav_path = Path(args.mix_wav)
    output_wav_path = Path(args.output_wav)
    save_mix_wav_path = Path(args.save_mix_wav)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not mix_wav_path.is_file():
        raise FileNotFoundError(f"mix wav not found: {mix_wav_path}")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    save_mix_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep a local copy of the source mix for direct A/B listening.
    shutil.copy2(mix_wav_path, save_mix_wav_path)

    device = _resolve_device(args.device)
    training_step = _load_model_for_inference(checkpoint_path, device)

    adapter = WaveformToSpectralBatchAdapter(compression_exponent=args.compression_exponent).to(device)

    ac_wave, bc_wave, sr = _read_mix_channels(
        mix_wav_path,
        ac_channel=args.ac_channel,
        bc_channel=args.bc_channel,
        sample_rate=args.sample_rate,
    )

    batch = {
        "mixture": ac_wave,
        "bc": bc_wave,
        "clean": ac_wave,
    }
    batch = {k: v.to(device) for k, v in batch.items()}

    autocast_enabled = args.use_amp and device.type == "cuda"
    autocast_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    with torch.no_grad():
        spectral_batch = adapter(batch).to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            model_output = training_step.model(spectral_batch.noisy_ac, spectral_batch.noisy_bc)

        enhanced_complex = complex_power_decompress(
            model_output.enhanced_complex,
            exponent=adapter.compression_exponent,
        )
        enhanced_wave = adapter.stft_processor.complex_to_waveform(
            enhanced_complex,
            length=ac_wave.shape[-1],
        )

    enhanced_np = enhanced_wave[0].detach().cpu().numpy()
    sf.write(str(output_wav_path), enhanced_np, sr, subtype="FLOAT")

    if args.save_input_ac is not None:
        ac_path = Path(args.save_input_ac)
        ac_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(ac_path), ac_wave[0].detach().cpu().numpy(), sr, subtype="FLOAT")

    if args.save_input_bc is not None:
        bc_path = Path(args.save_input_bc)
        bc_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(bc_path), bc_wave[0].detach().cpu().numpy(), sr, subtype="FLOAT")

    print(f"checkpoint: {checkpoint_path}")
    print(f"input: {mix_wav_path}")
    print(f"saved_mix: {save_mix_wav_path}")
    print(f"output: {output_wav_path}")
    print(f"device: {device}")
    print(f"length_samples: {enhanced_np.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
