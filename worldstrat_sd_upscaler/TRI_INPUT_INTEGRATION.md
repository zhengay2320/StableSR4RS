# Three-input Stage-3 integration

This document describes the implemented V1 path for processed RGB, an explicit
twelve-band Sentinel-2 raster, and an immutable offline ten-channel unmixing
array. The final model remains the existing RGB x4
`StableDiffusionUpscalePipeline`.

## Synthetic RGB input and target (current experiment)

`configs/stage3_tri_input.yaml` now follows the **data sources** of Stage 1:

```text
data_root/{train,val,test}/LR_bicubic/<name>  -> RGB model input
data_root/{train,val,test}/GT_geo_rad_visual/<name> -> RGB supervision/evaluation
tri_input.data.<split>.raw_ms_dir/<stem>.tiff -> auxiliary 12-band context
tri_input.data.<split>.unmixing_dir/<stem>.npy -> auxiliary offline F/U
```

RGB and GT are loaded from their own image files; neither is extracted from
the raw TIFF. Changing a raw observation must not change the loaded RGB/GT.
Missing RGB/GT is an error, not a request to construct it from spectral bands.
Nothing is regenerated, moved, or written back to these data directories.

The RGB parent is explicitly `outputs/stage1_synthetic/final` (LoRA and
ConditionAdapter), not the Stage-2 checkpoint. New results go to
`outputs/stage3_tri_input_synthetic`. The existing Stage-1 and Stage-2 YAMLs
are unchanged. Stage-3 still trains new modules by default; this change does
not silently enable training the frozen RGB parent or copy all Stage-1
optimizer hyperparameters.

Fixed `LR_bicubic` training is not random synthetic replay. The former keeps
same-scene raw/NPY as **auxiliary observations**, not as the physical observation
that generated the synthetic RGB. Their common spatial extent/grid must still
be verified: equal stems and dimensions alone cannot prove alignment, especially
after geometric GT correction. The checker only tests its internal structure
against raw spectra; it does not certify synthetic RGB or HR correctness.
Random replay remains disabled (`synthetic_replay_probability: 0.0`).

## Repository audit and scope

The implementation was audited against local commit
`bbe54a287920da88784c5e08a93cc94b06ef3a02`. This project is a Diffusers Stable
Diffusion x4 upscaler with UNet LoRA and a three-channel `ConditionAdapter`; it
is not the original StableSR SFT/time-aware encoder codebase and it is not SDXL.

The training path is:

1. `PairedSatelliteDataset` reads processed RGB LR and RGB GT into `[-1,1]`.
2. The frozen VAE encodes GT. The adapted RGB LR follows the existing low-res
   noise scheduler. The UNet contract remains `4 latent + 3 RGB` channels.
3. In tri-input mode, the same LR crop/flip/rotation is applied to raw and
   unmixing data. The condition uses original preprocessed RGB before low-res
   scheduler noise is sampled.
4. The conditioner produces `Q0`, checked `Q*`, four geometry scales, LR
   spectral context, and a warmup preview. A registered zero-initialized bridge
   projects them to runtime UNet down-block channels and sizes.
5. Fresh residual lists enter the pinned Diffusers public
   `down_intrablock_additional_residuals` argument. `conv_in`, scheduler,
   parameterization, LoRA layout, and ConditionAdapter are unchanged.
6. Training validation and whole/tiled inference use the same conditioner,
   bridge, and stateless UNet wrapper. A condition is computed once per sample
   or tile and reused for every denoising step.

Stage-3 defaults freeze existing LoRA, ConditionAdapter, VAE, text encoder, and
UNet base weights. Only the new conditioner and bridge train. Existing modules
can be explicitly unfrozen, but frozen forward passes are not put in
`no_grad`, so diffusion loss still reaches the residual condition.

## Data contract

Auxiliary files match by exact unique sample stem, never directory position.
Different `.png`, `.tiff`, and `.npy` extensions are supported. A CSV manifest
is the only supported renamed-file mapping. Duplicates, missing files, and grid
differences fail with the sample and full paths.

Rasterio reads raw TIFF lazily. Exactly twelve unique L2A names must be supplied
in stored order. All bands enter context; B1/B9 do not enter V1 surface
regression. Conversion must be explicit:

```yaml
band_names: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12] # example: verify TIFF
raw_value_conversion:
  mode: linear             # or identity if already in intended units
  scale: 0.0001            # example only: verify the producer convention
  offset: 0.0
  input_units: DN
  output_units: reflectance
```

There is no implicit divide-by-10000, per-patch stretch, or offset. Raster masks
and finite values define `raw_valid`; zero is not nodata. This is not cloud or
shadow detection, so real inputs must be screened upstream.

Offline arrays use `np.load(..., allow_pickle=False)` and must be finite float32
`[10,H,W]` in `[0,1]`. The first five channels are vegetation, water, bare,
snow, building fractions and the next five are `U`. They are not softmaxed or
renormalized. `F=0,U=1` means unknown. `U` is a weak use marker—not variance,
RMSE, correctness probability, or a cloud mask.
Low `U` can still be wrong, and the `building` channel is not a validated
roof-truth label.

Equal sample ID and dimensions are necessary but not proof of geographic
co-registration when RGB/NPY lack metadata. The production convention must be
recorded in the precheck report.

## Precheck and train-only statistics

Fill verified band order/conversion in `configs/stage3_tri_input.yaml`. Keep
unknown val/test paths `null`; never guess them.

```bash
python scripts/validate_tri_input_data.py \
  --config configs/stage3_tri_input.yaml --split train \
  --output-dir outputs/tri_input_data_check

python scripts/compute_raw_band_stats.py \
  --config configs/stage3_tri_input.yaml \
  --output outputs/tri_input_data_check/raw_band_stats.json
```

Auxiliary directories remain the explicit per-split paths in your YAML; they
are independent of `data_root` and are not changed by selecting `LR_bicubic`.
A directory name does not redefine its configured split. Point
`tri_input.raw_stats_path` to the generated JSON. Val/test/inference load those
train-only statistics.

For val/test, supply exact confirmed paths:

```bash
python scripts/validate_tri_input_data.py \
  --config configs/stage3_tri_input.yaml --split val \
  --rgb-dir /exact/processed/val/LR_bicubic \
  --raw-ms-dir /exact/raw/val \
  --unmixing-dir /exact/unmixing/val \
  --output-dir outputs/tri_input_data_check
```

## A/B warmup and diffusion adaptation

Stage A disables the checker and does not load the diffusion model:
Both warmup stages read `train/LR_bicubic` and `train/GT_geo_rad_visual`
from the current configuration, just like diffusion adaptation.

```bash
python scripts/train_tri_input_warmup.py \
  --config configs/stage3_tri_input.yaml --stage A \
  --output_dir outputs/tri_input_warmup_a --max_steps 2000
```

Stage B loads A's weights, enables the checker, and continues preview warmup:

```bash
python scripts/train_tri_input_warmup.py \
  --config configs/stage3_tri_input.yaml --stage B \
  --init_conditioner outputs/tri_input_warmup_a/final \
  --output_dir outputs/tri_input_warmup_b --max_steps 2000
```

Set `tri_input.init_conditioner_path` to Stage B `final`, set confirmed val
paths, then run Stage C:

```bash
accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  src/train_lora_upscaler.py --config configs/stage3_tri_input.yaml
```

`resume_from_checkpoint` is only for the same Stage-3 structure. RGB init uses
explicit `init_lora_path`/`init_adapter_path` and never forces an old optimizer
state into new parameter groups. A declared tri-input checkpoint missing either
new weight file fails immediately.

Random synthetic RGB replay defaults to zero. A nonzero value plus policy `error`
rejects false pairing. Explicit policy `disable` bypasses the complete new
branch and preview loss for replayed samples; it never retains real raw data
while randomly replacing only RGB. This does not disable auxiliaries for the
explicit fixed `LR_bicubic` experiment described above.

## Validation and inference

Training validation uses the same condition and wrapper. Independent validation
uses the inference entry with GT, followed by the existing evaluator:

```bash
python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path outputs/stage3_tri_input_synthetic/final \
  --input_dir /exact/processed/val/LR_bicubic \
  --gt_dir /exact/processed/val/GT_geo_rad_visual \
  --split val --raw_ms_dir /exact/raw/val \
  --unmixing_dir /exact/unmixing/val \
  --output_dir outputs/stage3_tri_input_synthetic_val

python src/evaluate.py \
  --sr_dir outputs/stage3_tri_input_synthetic_val/sr_raw \
  --gt_dir outputs/stage3_tri_input_synthetic_val/gt \
  --lr_dir /exact/processed/val/LR_bicubic \
  --output_dir outputs/stage3_tri_input_synthetic_val_metrics --device auto
```

GT-free whole-image test:

```bash
python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path outputs/stage3_tri_input_synthetic/final \
  --input_dir /exact/processed/test/LR_bicubic \
  --split test --raw_ms_dir /exact/raw/test \
  --unmixing_dir /exact/unmixing/test \
  --output_dir outputs/stage3_tri_input_synthetic_test
```

Tiled inference adds `--tiled --tile_size 128 --tile_overlap 32`. All three
inputs use identical LR coordinates. RGB edge-pads; raw pads zero with invalid
mask; prior pads `F=0,U=1`. Padding is excluded from scoring. A small image
explicitly takes whole-image inference without resizing. Hann blending, x4
cropping, `sr_raw`, `sr_projected`, and low-frequency projection alpha remain.

Use `--disable_tri_input` only for a deliberate RGB-only ablation. Otherwise,
missing auxiliary data or weights is an error. Old artifacts without a tri
declaration retain the old RGB-only path and need no auxiliary inputs.

## Checkpoints and scientific limits

Stage C adds `tri_input_conditioner.safetensors`, `tri_input_config.json`,
`tri_input_bridges.safetensors`, `tri_input_bridge_config.json`,
`raw_band_stats.json`, and per-rank RNG state while retaining existing files.
The artifact records exact bands, conversion, stats, checker, injection layout,
stage, and RGB parent. Loads are strict.

V1 tests only stationary/left/right/up/down candidates; one unit is one target
pixel. It refits a shared local spectral matrix per candidate using two
interleaved folds and float32 ridge solves. It does not calibrate a sensor PSF,
recover physical HR non-RGB spectra, rerun FCLS, estimate a posterior, or
validate semantic classes. Average pooling is a common-grid approximation.
`scores` and `support` are internal heuristics—not accuracy, probability,
calibrated confidence, or independent truth. Ambiguous, under-supported,
ill-conditioned, or non-improving cases remain stationary.
