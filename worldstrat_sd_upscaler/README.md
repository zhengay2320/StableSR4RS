# WorldStrat Stable Diffusion ×4 Upscaler

Reproducible two-stage LoRA fine-tuning of the dedicated Hugging Face
`StableDiffusionUpscalePipeline` (`stabilityai/stable-diffusion-x4-upscaler`) for paired
WorldStrat remote-sensing super-resolution. The project pins the source checkout to Diffusers
`v0.39.0`; it does not use `StableDiffusionPipeline` or SDXL.

The original `GT`, `LR`, and `LR_bicubic` trees are read-only inputs. Training, validation,
inference, data-quality logs, checkpoints, and metrics are written beneath this project's
`outputs/` directory.

## 1. Create the environment

Use Python 3.10 or 3.11 and install the PyTorch build matching the host CUDA version first.
The setup script deliberately refuses to guess or replace PyTorch.

```bash
cd worldstrat_sd_upscaler
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the correct torch + torchvision build from https://pytorch.org/get-started/locally/
bash scripts/setup_env.sh
```

`setup_env.sh` clones `huggingface/diffusers` at the exact shallow tag `v0.39.0` into
`third_party/diffusers`, installs it editable, then installs the remaining dependencies.
`bitsandbytes` and `xformers` are optional and are never installed implicitly.

## 2. Hugging Face access

Accept the model license on its Hub page if requested, then authenticate without putting a token
in source code:

```bash
hf auth login
# or set HF_TOKEN only in the shell/session or secret manager
```

The first model load downloads `stabilityai/stable-diffusion-x4-upscaler` into the normal Hugging
Face cache.

## 3. Validate the data

The default root is already present in both YAML files and scripts:

```bash
export DATA_ROOT=/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image
bash scripts/validate_data.sh
```

Validation uses exact filenames, verifies image readability and `GT = 4 × LR` geometry, and writes
bad records to `outputs/data_validation/*.csv`. Invalid pairs are skipped by default. Set
`strict_pairs: true` in a YAML file to stop on the first invalid record. Images smaller than the
configured crop fail explicitly and are never enlarged.

To validate exactly one training sample:

```bash
python src/validate_dataset.py --data_root "$DATA_ROOT" --split train \
  --lr_subdir LR_bicubic --max_samples 1 \
  --output_csv outputs/data_validation/one_sample.csv
```

## 4. Run the smoke test

The smoke script compiles the package, runs unit tests, validates one real pair, performs two
optimizer steps, saves and reloads LoRA/ConditionAdapter artifacts through inference, generates one
test image, and asserts ×4 output geometry:

```bash
DATA_ROOT="$DATA_ROOT" bash scripts/smoke_test.sh
```

It still requires access to the base model and enough accelerator memory. The unit tests do not
download the model.

## 5. Stage 1 — same-source synthetic pretraining

Stage 1 trains on `train/LR_bicubic → train/GT` and writes
`outputs/stage1_synthetic/{checkpoint-*,final}`:

```bash
bash scripts/train_stage1.sh
```

## 6. Stage 2 — cross-sensor transfer

Stage 2 loads the final Stage 1 LoRA and ConditionAdapter, trains mainly on
`train/LR → train/GT`, and replays same-name `LR_bicubic` with configurable probability
`synthetic_replay_probability: 0.25`:

```bash
bash scripts/train_stage2.sh
```

Every artifact directory contains:

```text
pytorch_lora_weights.safetensors
condition_adapter.safetensors
training_config.yaml
model_info.json
optimizer.pt                 # checkpoints/final training state
lr_scheduler.pt
trainer_state.pt
```

LoRA uses PEFT on the UNet attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`) only and is
saved through the Diffusers pipeline API, so it can be loaded with
`StableDiffusionUpscalePipeline.load_lora_weights()`. VAE and CLIP text encoder remain frozen;
tokenizer and both schedulers have no optimizer parameters.

## 7. Synthetic test inference

```bash
bash scripts/infer_synthetic_test.sh
```

This evaluates Stage 1 on `test/LR_bicubic`. Outputs retain the input filename and are separated
into `sr_raw/`, `sr_projected/`, `lr_bicubic/`, `gt/`, and `previews/`.

Inference can load any saved artifact directory, not only `final`. The directory must contain
both `pytorch_lora_weights.safetensors` and `condition_adapter.safetensors`:

```bash
python src/infer_upscaler.py --config configs/stage1_synthetic.yaml \
  --checkpoint_path outputs/stage1_synthetic/checkpoint-01000 \
  --input_dir "$DATA_ROOT/test/LR_bicubic" --gt_dir "$DATA_ROOT/test/GT" \
  --output_dir outputs/inference_stage1_checkpoint_01000
```

Select individual samples by exact filename or filename stem. Repeat `--sample` to select several,
or use `--sample_file` with one filename/stem per line:

```bash
python src/infer_upscaler.py --config configs/stage1_synthetic.yaml \
  --checkpoint_path outputs/stage1_synthetic/checkpoint-01000 \
  --input_dir "$DATA_ROOT/test/LR_bicubic" --gt_dir "$DATA_ROOT/test/GT" \
  --sample sample_001.png --sample sample_017 \
  --output_dir outputs/inference_selected_samples

python src/infer_upscaler.py --config configs/stage1_synthetic.yaml \
  --checkpoint_path outputs/stage1_synthetic/checkpoint-01000 \
  --input_dir "$DATA_ROOT/test/LR_bicubic" --sample_file samples.txt \
  --output_dir outputs/inference_sample_list
```

## 8. Real test inference

```bash
bash scripts/infer_real_test.sh
```

This evaluates Stage 2 on real `test/LR`. Defaults are `guidance_scale=1.0`, `noise_level=10`,
`num_inference_steps=40`, seed 42, the restrained fixed geographic prompt, and low-frequency
projection alpha 0.5. The optional negative prompt is used only when guidance exceeds 1.

The checkpoint and inference data source are independent. For example, evaluate a Stage 1 model
trained with synthetic bicubic LR on the real LR observations:

```bash
python src/infer_upscaler.py --config configs/stage1_synthetic.yaml \
  --checkpoint_path outputs/stage1_synthetic/checkpoint-01000 \
  --input_dir "$DATA_ROOT/test/LR" --gt_dir "$DATA_ROOT/test/GT" \
  --output_dir outputs/inference_stage1_checkpoint_01000_on_real
```

The wrapper scripts also accept overrides after their defaults, for example:

```bash
bash scripts/infer_synthetic_test.sh \
  --checkpoint_path outputs/stage1_synthetic/checkpoint-01000 \
  --input_dir "$DATA_ROOT/test/LR" \
  --output_dir outputs/inference_stage1_checkpoint_01000_on_real \
  --sample sample_001.png
```

For large images, smooth overlapping tiling uses LR tiles 128, overlap 32, corresponding HR tiles
512, and a 2-D Hann blend:

```bash
bash scripts/infer_real_test.sh --tiled --tile_size 128 --tile_overlap 32
```

Projection ablations always preserve raw SR and can be selected with:

```bash
bash scripts/infer_real_test.sh --low_freq_projection_alpha 0
bash scripts/infer_real_test.sh --low_freq_projection_alpha 0.5
bash scripts/infer_real_test.sh --low_freq_projection_alpha 1.0
```

## 9. Metrics

```bash
bash scripts/evaluate_synthetic.sh
bash scripts/evaluate_real.sh
```

Evaluation produces `per_image_metrics.csv`, `summary_metrics.json`, and `summary_metrics.csv`.
Metrics are PSNR, SSIM, LPIPS, RGB MAE, R/G/B bias, RGB spectral angle in degrees, and LR
observation consistency after bicubic SR downsampling (MAE, PSNR, spectral angle). Summary files
contain mean, median, population standard deviation, P5, and P95. If LPIPS is unavailable, the
evaluator logs a clear installation hint and writes the other metrics; `--skip_lpips` disables it
explicitly.

### Untouched base-model benchmark on both test conditions

`scripts/evaluate_unfinetuned_base.py` is a self-contained inference-and-evaluation benchmark for
the original pretrained Stable Diffusion x4 Upscaler. It deliberately loads no project LoRA,
ConditionAdapter, or latent phi. The model is loaded once and then evaluated on every valid pair
from both `test/LR_bicubic` (synthetic) and `test/LR` (real), against
`test/GT_geo_rad_visual`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_unfinetuned_base.py
```

The model/data/output paths and inference settings are fixed constants at the top of the script.
Each condition receives `sr_raw/`, optional previews, per-image metrics, and summary CSV/JSON;
`outputs/unfinetuned_base_benchmark/summary_comparison.csv` contains the side-by-side means. The
script performs generation and scoring in the same loop, so no separate inference or evaluator
command is needed.

### Cas-All checkpoint without the Cas inference branch

`scripts/evaluate_cas_all_backbone_only.py` is the matching ablation for a trained Cas-All
checkpoint. It loads that checkpoint's UNet LoRA and ConditionAdapter, but deliberately does not
load or call latent phi or learned `r_t`. Inference therefore uses the original pipeline scheduler
path while retaining the theta/adapter weights learned during Cas-All training. It evaluates every
valid synthetic and real test pair and computes the metrics in the same process:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cas_all_backbone_only.py
```

Fixed paths and inference settings are declared at the top of the script. Results are written to
`outputs/cas_all_backbone_only_benchmark/`; the main comparison is
`summary_comparison.csv`, with per-condition images and metrics under `synthetic/` and `real/`.
The per-sample seed rule matches `infer_stage1_latent_phi.py` (`seed + sorted test index`) so this
run can be compared against full Cas-All inference without changing the diffusion noise.

## 10. Resume a checkpoint

Resume restores LoRA, ConditionAdapter, optimizer, LR scheduler, step, and PyTorch RNG state:

```bash
accelerate launch src/train_lora_upscaler.py \
  --config configs/stage1_synthetic.yaml \
  --resume_from_checkpoint outputs/stage1_synthetic/checkpoint-01000

# Resolve the numerically newest checkpoint in the configured output directory:
accelerate launch src/train_lora_upscaler.py \
  --config configs/stage1_synthetic.yaml --resume_from_checkpoint latest
```

`checkpoints_total_limit` controls retention. Only the main process writes artifacts.

## 11. Single-GPU training

Python never hardcodes GPU count. Select a device in the shell and use ordinary Accelerate launch:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  src/train_lora_upscaler.py --config configs/stage1_synthetic.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  src/train_lora_upscaler.py --config configs/stage2_cross_sensor.yaml
```

## 12. Four-GPU training

The stage scripts default to devices `0,1,2,3` and exactly four processes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  src/train_lora_upscaler.py --config configs/stage1_synthetic.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  src/train_lora_upscaler.py --config configs/stage2_cross_sensor.yaml
```

Accelerate prepares the UNet, ConditionAdapter, optimizer, scheduler, DataLoader, and distributed
sampler. `mixed_precision` accepts `fp16`, `bf16`, or `no`; accumulation is set in YAML.

## 13. If GPU memory is insufficient

- Keep `train_batch_size: 1`; reduce it only if a larger custom value was selected.
- Reduce `gt_crop_size` in multiples of 32 while preserving divisibility by 4. This changes the LR
  crop to `gt_crop_size / 4`; 512/128 is the intended default.
- Increase `gradient_accumulation_steps` to preserve effective batch size.
- Keep `gradient_checkpointing: true`.
- Install `xformers` matching PyTorch/CUDA, then set
  `enable_xformers_memory_efficient_attention: true`.
- Optionally install `bitsandbytes` and set `use_8bit_adam: true`.
- Reduce validation sample count or validation inference steps if validation is the peak.

## 14. Controlled comparisons

Use identical test files, prompts, seeds, noise levels, steps, and metric commands:

1. **Base model:** run `infer_upscaler.py` without `--lora_path` and `--adapter_path` (the adapter is
   then a zero-initialized identity).
2. **Stage 1:** use `outputs/stage1_synthetic/final` and synthetic test LR.
3. **Stage 2:** use `outputs/stage2_cross_sensor/final`, real test LR, and evaluate `sr_raw/`.
4. **Stage 2 + projection:** evaluate `sr_projected/` at alpha 0.5; repeat with 0 and 1 for the full
   ablation.

Example base invocation:

```bash
python src/infer_upscaler.py --config configs/stage2_cross_sensor.yaml \
  --input_dir "$DATA_ROOT/test/LR" --gt_dir "$DATA_ROOT/test/GT" \
  --output_dir outputs/inference_base --low_freq_projection_alpha 0
```

## Configuration and reproducibility

All YAML values are consumed by the implementation: model/data paths, crop and replay policy,
LoRA and adapter optimization, noise ranges, prompt/dropout behavior, precision and memory options,
checkpoint/validation cadence, SNR gamma, and seed. Actual resolved configuration is copied into
each run and artifact directory. Metadata prompts are optional: when `final_metadata.csv` or its
required fields are absent, the fixed prompt is used without interrupting training.

## Clean-latent reliability across diffusion timesteps

The post-training diagnostic below fixes one aligned synthetic validation pair, one VAE latent
sample, one diffusion noise tensor, and one noisy LR condition while evaluating every training
timestep. It converts the scheduler's actual `epsilon`, `v_prediction`, or clean-sample output to
an estimated clean latent, decodes both `z_t` and the estimated clean latent with the pipeline VAE,
and writes full metrics plus representative visualizations:

```bash
python scripts/analyze_z0_reliability.py \
  --config stage1_synthetic.yaml \
  --checkpoint outputs/stage1_synthetic/final \
  --num_samples 1 --noise_seed 1234 --timestep_stride 1
```

Results default to `outputs/z0_reliability/stage1_synthetic_final/`. Use a larger
`--timestep_stride` for a quick pipeline check. LPIPS is used when its optional dependency and
backbone are available; pass `--skip_lpips` for an entirely offline run.

The same command also runs a second, inference-faithful trajectory from a Gaussian latent that is
independent of HR. It records the formula-derived clean prediction before every scheduler step in
`pure_noise_trajectory/metrics.csv` and `pure_noise_trajectory_grid.png`. In that trajectory HR is
used only after prediction as a visual/metric reference. To disable the HR-derived forward-noising
probe and run only this pure-noise path, add `--skip_training_timestep_analysis`; control the
trajectory length with `--num_inference_steps` (default: the validation setting in the YAML).

## Stage 1 learned-r_t latent Cas-DM timestep ablation

The two Cas-DM configurations keep Stage 1 data, seed, optimization, prompts, and validation
settings identical. The latent CNN predicts both a refined clean latent and a single-channel
spatial `r_t`. Training uses clean-latent MSE, differentiable decoded LPIPS, and the paper's
stop-gradient reverse-mean mixture loss `L_mu`. The inclusive ranges use
`0 = low noise / reverse late` and `1 = high noise / reverse early`; no range is learned.

Baseline, Cas-All, and Cas-Late20 training commands are:

```bash
accelerate launch src/train_lora_upscaler.py --config configs/stage1_synthetic.yaml
accelerate launch src/train_lora_upscaler.py --config configs/stage1_synthetic_cas_all.yaml
accelerate launch src/train_lora_upscaler.py --config configs/stage1_synthetic_cas_late20.yaml
```

Enabled checkpoints additionally contain `latent_phi.safetensors` and
`latent_phi_config.json`; the existing `optimizer.pt` includes the phi parameter-group state.
Configs without `phi_enabled: true` retain the original checkpoint contents and training path.

Run each complete Cas-DM checkpoint with its matching range:

```bash
python src/infer_stage1_latent_phi.py \
  --config configs/stage1_synthetic_cas_all.yaml \
  --checkpoint outputs/stage1_synthetic_cas_all/final \
  --output_dir outputs/cas_all_test

python src/infer_stage1_latent_phi.py \
  --config configs/stage1_synthetic_cas_late20.yaml \
  --checkpoint outputs/stage1_synthetic_cas_late20/final \
  --output_dir outputs/cas_late20_test
```

This independent Cas-DM inference entry defaults to the complete `test` split and does not impose
a sample limit. `--limit N` (or the legacy alias `--num_samples N`) is only for debugging. Use
`--split val` explicitly for validation. Results preserve the original filenames under `sr_raw/`,
`gt/`, and `lr_input/`, so the common evaluator can process every pair directly:

```bash
python src/evaluate.py \
  --sr_dir outputs/cas_all_test/sr_raw \
  --gt_dir outputs/cas_all_test/gt \
  --lr_dir outputs/cas_all_test/lr_input \
  --output_dir outputs/evaluation_cas_all_test

python src/evaluate.py \
  --sr_dir outputs/cas_late20_test/sr_raw \
  --gt_dir outputs/cas_late20_test/gt \
  --lr_dir outputs/cas_late20_test/lr_input \
  --output_dir outputs/evaluation_cas_late20_test
```

For a baseline generated with the identical test loader, seed, DDIM steps, and noise level, use
the same independent entry with `--disable_phi`:

```bash
python src/infer_stage1_latent_phi.py \
  --config configs/stage1_synthetic.yaml \
  --checkpoint outputs/stage1_synthetic/final \
  --disable_phi \
  --output_dir outputs/baseline_stage1_test
```

At an active DDIM timestep (`eta=0`), base and refined model outputs are stepped independently
from the same current latent and their two `prev_sample` means are mixed by learned spatial
`r_t`. At an inactive timestep, the script makes exactly one unchanged baseline scheduler step.

The optional z0 diagnostic accepts the same checkpoint without changing its baseline behavior
when `--phi_path` is omitted:

```bash
python scripts/analyze_z0_reliability.py \
  --config configs/stage1_synthetic_cas_late20.yaml \
  --checkpoint outputs/stage1_synthetic_cas_late20/final \
  --phi_path outputs/stage1_synthetic_cas_late20/final \
  --phi_timestep_range 0.0 0.2 \
  --num_samples 1 --timestep_stride 100 --skip_lpips
```
