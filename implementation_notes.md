# Implementation Notes

This document separates what is explicitly specified in the paper from engineering assumptions used for runnable code.

## Paper-Specified Items Implemented in Phase 2/3
- Complex spectrum representation uses two channels: real + imag, with canonical shape `[B, 2, T, 161]`.
- STFT parameters: sample rate 16 kHz, window length 20 ms, hop 10 ms, Hann window, FFT size 320, single-sided `F=161`.
- iAFF two-stage fusion equations:
  - `a0 = Fa1(y_AC + y_BC)`
  - `y_AF' = a0 * y_AC + (1 - a0) * y_BC`
  - `a = Fa2(y_AF')`
  - `y_AF = a * y_AC + (1 - a) * y_BC`
- Dense-layer-level conv setup: kernel `(1,3)`, stride `(1,1)`, output channels `8`, each conv followed by BN + PReLU, with DenseNet-style concatenation.
- Gated layer mechanism: main branch + sigmoid gate branch, element-wise multiplication, then BN + PReLU (with option to disable for decoder first block special case).
- Decoder cRM head: split 2 channels into real/imag and pass each through frequency-wise FC `161 -> 161`.

## Engineering Assumptions Used in Phase 2/3
- iAFF channel attention internals (local/global context tensorization and hidden channel size) are not fully specified; implemented via 1x1 local path + GAP global path.
- Dense conv padding for in-block shape preservation uses `(0,1)`.
- Gated conv/deconv padding uses `(0,0)` to match Table I frequency progression.
- Deconvolution `output_padding` selection for exact recovery to 161 bins follows:
  - `3->8`: `(0,0)`
  - `8->18`: `(0,0)`
  - `18->38`: `(0,0)`
  - `38->79`: `(0,1)`
  - `79->161`: `(0,1)`
- STFT implementation details not explicitly specified in paper:
  - `center=True`
  - `normalized=False`
- Attention Gate internals not specified; implemented as additive attention gate (Attention U-Net style) with optional interpolation for size alignment.

## Deferred Assumptions (To Be Implemented in Later Phases)
- Grouped sConformer complete internals (placeholder only, replaceable interface planned).
- AG wiring details across all encoder/decoder levels in full DenGCAN integration.
