# 三路输入合成训练推理与评估完整操作手册

本手册对应当前 `worldstrat_sd_upscaler` 实现，整理从数据预检、Stage 1 RGB 模型、三路预热、扩散适配到完整测试集推理与评估的操作顺序。所有命令都是 Linux Bash 命令，路径直接填写，不使用路径变量、参数数组或占位路径。

请逐节执行，每个阶段确认成功后再进行下一个阶段，不要将整篇命令一次性全部粘贴执行。已完成且配置未改变的阶段可以跳过。长任务使用 `nohup`，单张 GPU 上按顺序运行，不要同时启动 A、B、正式训练和推理。

## 1 当前实验和调用关系

RGB 输入为已经准备好的 `LR_bicubic` 图片，RGB 目标为已经准备好的 `GT_geo_rad_visual` 图片。两者都不从多光谱 TIFF 提取，也不在本流程中重新生成。原始 TIFF 和离线 NPY 只作同场景辅助条件，最终仍输出 RGB 四倍超分图。

```text
LR_bicubic RGB ──→ 原 ConditionAdapter ──→ 低分辨率条件加噪 ───────────────┐
                                                                       │
LR_bicubic RGB ─┐                                                       │
12 波段 TIFF ──┼→ 三路编码与融合 → Q0 → 五候选光谱检查 → Q* → 多尺度桥接 ┤
离线 F/U NPY ──┘                                                       │
                                                                       ↓
训练时 GT → 冻结 VAE 编码 → latent 加噪 ───────────────────────→ 原 UNet
推理时纯噪声 latent ──────────────────────────────────────────→ 反向采样
                                                                       ↓
                                                               VAE 解码 → RGB ×4
```

GT 只用于训练监督和事后评价，不进入推理条件。新模块在低分辨率条件随机加噪前读取 RGB；条件每个样本或 tile 计算一次，在去噪各步复用。

当前训练阶段如下。

| 阶段 | 输入与目标 | 更新的部分 | 用途 |
|---|---|---|---|
| Stage 1 | RGB LR_bicubic → RGB GT_geo_rad_visual | UNet LoRA、原 ConditionAdapter | 得到已有 RGB 超分模型，已有合格 checkpoint 可跳过 |
| 预热 A | 三路输入 → RGB preview，与 GT 比较 | 三路编码器、融合、上采样、preview head | 不加载扩散模型，checker 关闭 |
| 预热 B | 与 A 相同，checker 开启 | 与 A 相同 | 继续 preview 预热并记录 checker 内部诊断 |
| 正式三路训练 C | 三路条件与原 diffusion 目标 | 新 conditioner 的有效梯度分支、桥接层 | 从 Stage 1 RGB 权重和 B 预热权重初始化 |
| 推理与测试 | RGB LR_bicubic 加对应 raw/NPY | 不更新参数 | 生成 RGB ×4，GT 仅事后评价 |

重要实现边界：当前 `preview = preview_head(hr_features)` 不经过 `Q0 → checker → Q* → geometry`，预热损失只有 preview L1。A/B 不通过这个损失训练 layout head 或 geometry 分支；B 的预览图不能证明 checker 改善。正式 C 的 diffusion loss 才通过 bridge、geometry、Q* 回传到相应分支。checker 的离散选择本身不反传，最终结构 warp 保留梯度。

C 默认冻结已有 LoRA、ConditionAdapter、UNet 主参数、VAE 和 text encoder。`phi_enabled=false`、`rgb_aux_loss_enabled=false`、`preview_loss_weight=0`；C 使用原 Min-SNR diffusion loss，不启用 Cas-DM 或直接 RGB 辅助损失。

本流程不需要 Stage 2 cross-sensor 训练。固定 `LR_bicubic` 主输入不是随机 synthetic replay；回放概率保持 0。真实 raw/NPY 是场景辅助观测，不应宣称它们与合成 RGB 来自同一次退化。

## 2 目录和配置核对

服务器项目根目录：

```text
/data/zhengay/StableSR/worldstrat_sd_upscaler
```

以下原始目录均只读使用，不移动、覆盖或重新计算其中的数据：

| 内容 | 当前配置路径 |
|---|---|
| RGB 数据根目录 | `/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image` |
| train 原始多光谱 | `/data/zhengay/EDiffSR-main/data/new_star/train/lr` |
| train 离线解混 | `/data/zhengay/EDiffSR-main/data/new_star/train/lr_unmixing` |
| val 原始多光谱 | `/data/zhengay/EDiffSR-main/data/new_star/val/lr` |
| val 离线解混 | `/data/zhengay/EDiffSR-main/data/new_star/val/lr_unmixing` |
| test 原始多光谱 | `/data/zhengay/EDiffSR-main/data/new_star/test/lr` |
| test 离线解混 | `/data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing` |

这些是当前 YAML 值，不是本地已经核验过的服务器路径。开始前必须解决一处路径冲突：你最早明确提供的 train 解混路径是 `/data/zhengay/EDiffSR-main/data/new_star/test/train_unmixing`，与当前 YAML 不同。本次只整理命令，不擅自改回历史配置；在确认当前 train raw 究竟对应哪份离线结果之前，不要执行统计、预热或训练。如果原先提供的目录才是实际对应目录，必须先将 `tri_input.data.train.unmixing_dir` 改为该路径，并同步修改下文 train 预检命令；不能只覆盖预检参数而让统计、预热和训练继续读取另一目录。不能仅凭目录存在或位于 train/test 下决定配对。val/test 也必须按实际数据核实，不能按名称猜测。

在 `configs/stage3_tri_input.yaml` 中核对以下已有字段，不要重复添加第二个 `tri_input` 节点：

```yaml
model_id: /data/zhengay/models/stabilityaistable-diffusion-x4-upscaler
data_root: /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image
train_lr_subdir: LR_bicubic
val_lr_subdir: LR_bicubic
test_lr_subdir: LR_bicubic
gt_subdir: GT_geo_rad_visual
output_dir: outputs/stage3_tri_input_synthetic
init_lora_path: outputs/stage1_synthetic/final
init_adapter_path: outputs/stage1_synthetic/final
synthetic_replay_probability: 0.0
phi_enabled: false
rgb_aux_loss_enabled: false

tri_input:
  enabled: true
  training_stage: diffusion
  train_existing_lora: false
  train_existing_condition_adapter: false
  synthetic_aux_policy: error
  preview_loss_weight: 0.0
  raw_stats_path: outputs/tri_input_data_check/raw_band_stats.json
  init_conditioner_path: outputs/tri_input_synthetic_warmup_b/final
  band_names: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12]
  raw_value_conversion:
    mode: identity
    scale: 1.0
    offset: 0.0
    input_units: reflectance
    output_units: reflectance
```

这只是关键字段摘录，不是替换整个配置文件的完整 YAML；保留原有 `data`、`checker` 等字段。上面的统计路径在第 5 节产生，预热路径在第 8 节产生，在它们产生之前不要启动 C。

波段顺序与 identity 对应当前离线代码约定及本地抽查的浮点 TIFF。服务器若使用不同制作版本，需重新确认，不能仅凭 12 通道判断顺序，也不能重复除以 10000。已训练 checkpoint 的统计和转换契约不能随意替换。

## 3 环境准备和日志目录

先同步最新代码，包括已修复 fp16 残差精度的 `src/tri_input_bridge.py` 和 `src/train_lora_upscaler.py`。只需同步代码，不重新运行离线解混，不重新下载基础模型。

```bash
conda activate diffusion_stable
cd /data/zhengay/StableSR/worldstrat_sd_upscaler
mkdir -p /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs
```

```bash
python -c "import torch, rasterio, accelerate, diffusers; print('torch:', torch.__version__); print('diffusers:', diffusers.__version__); print('CUDA:', torch.cuda.is_available()); print('rasterio:', rasterio.__version__)"
```

当前实现要求 pinned Diffusers 0.39.0。不要为运行本手册随意升级 PyTorch 或 Diffusers。缺依赖时先解决对应依赖，再继续；不需要重跑会下载模型的流程。

下文 GPU 任务均在命令开头直接指定 `CUDA_VISIBLE_DEVICES=1`，表示物理 GPU 1；只有 GPU 0 时，将每条命令中的数字改为 0。这不是路径变量，不需要额外 export。Hub 离线开关用于避免意外下载基础模型；缺本地组件时应修复模型目录。

`nohup` 返回后任务仍在运行，不能立刻开始下一阶段。通过日志确认正常结束且有新的 final 保存记录；旧的 final 目录单独存在不能证明本次任务完成。`tail -f` 用 Ctrl+C 退出只停止看日志，不会停止后台任务。不要在同一输出目录同时运行两个任务，也不要用旧结果目录混装不同实验。

## 4 预检 RGB 标签和三路配对

### 4.1 RGB 与 GT 的严格四倍关系

三路预检不检查 HR，因此先分别检查三个 split 的 RGB 和 GT：

```bash
python src/validate_dataset.py \
  --data_root /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image \
  --split train --lr_subdir LR_bicubic --gt_subdir GT_geo_rad_visual \
  --scale 4 --strict_pairs \
  --output_csv /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check/train_invalid_rgb_pairs.csv
```

```bash
python src/validate_dataset.py \
  --data_root /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image \
  --split val --lr_subdir LR_bicubic --gt_subdir GT_geo_rad_visual \
  --scale 4 --strict_pairs \
  --output_csv /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check/val_invalid_rgb_pairs.csv
```

```bash
python src/validate_dataset.py \
  --data_root /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image \
  --split test --lr_subdir LR_bicubic --gt_subdir GT_geo_rad_visual \
  --scale 4 --strict_pairs \
  --output_csv /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check/test_invalid_rgb_pairs.csv
```

### 4.2 RGB 与 raw TIFF 和 NPY 的配对

```bash
python scripts/validate_tri_input_data.py \
  --config configs/stage3_tri_input.yaml --split train \
  --rgb-dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/train/LR_bicubic \
  --raw-ms-dir /data/zhengay/EDiffSR-main/data/new_star/train/lr \
  --unmixing-dir /data/zhengay/EDiffSR-main/data/new_star/train/lr_unmixing \
  --output-dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check
```

```bash
python scripts/validate_tri_input_data.py \
  --config configs/stage3_tri_input.yaml --split val \
  --rgb-dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/val/LR_bicubic \
  --raw-ms-dir /data/zhengay/EDiffSR-main/data/new_star/val/lr \
  --unmixing-dir /data/zhengay/EDiffSR-main/data/new_star/val/lr_unmixing \
  --output-dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check
```

```bash
python scripts/validate_tri_input_data.py \
  --config configs/stage3_tri_input.yaml --split test \
  --rgb-dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --raw-ms-dir /data/zhengay/EDiffSR-main/data/new_star/test/lr \
  --unmixing-dir /data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing \
  --output-dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check
```

完成标志是三个 split 均成功，报告中的 `failure_count` 为 0。输出包括各 split 的配对 manifest 和 JSON 报告。原 RGB/GT 以同文件名配对，辅助数据以唯一 stem 或显式 manifest 配对，不能按排序位置对应。若目录路径有修改，先回写 YAML 再继续。

预检能验证读取契约、尺寸和配对，不能凭同名同尺寸证明真实地理对齐。尤其 GT 已经过几何校正，必须确认合成 RGB 与辅助 raw/NPY 对应的范围和网格制作约定。

## 5 计算训练集原始波段统计

```bash
python scripts/compute_raw_band_stats.py \
  --config configs/stage3_tri_input.yaml \
  --rgb-dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/train/LR_bicubic \
  --output /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_data_check/raw_band_stats.json
```

真实实验不加 `--limit`。这个脚本始终读取 YAML 中的 train raw/NPY 路径，不支持另传 `--raw-ms-dir` 或 `--unmixing-dir`。检查统计覆盖的样本数及有效像元数，再确认 YAML 的 `tri_input.raw_stats_path` 指向此文件。

val/test 不重新统计。训练保存的三路 checkpoint 会包含 `raw_band_stats.json`；推理默认使用该 checkpoint 内的统计，并检查与模型缓冲区一致。若已经有相同输入集合、转换方式和波段顺序的正确统计，可以直接复用，不要在已训练模型上随意替换统计。

## 6 Stage 1 RGB 模型准备

已完成 Stage 1 时跳过训练，先检查两个文件：

```bash
ls -l /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage1_synthetic/final/pytorch_lora_weights.safetensors
ls -l /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage1_synthetic/final/condition_adapter.safetensors
```

仅在没有已有合格 Stage 1 权重时运行下面命令。该训练不启用三路功能，使用原配置，当前为 20000 步。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  src/train_lora_upscaler.py \
  --config configs/stage1_synthetic.yaml \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage1_synthetic \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/stage1_rgb_train.log 2>&1 < /dev/null &
```

```bash
tail -f /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/stage1_rgb_train.log
```

已有 Stage 1 权重时不要为了接入三路重新训练或覆盖它。C 的 YAML `init_lora_path` 和 `init_adapter_path` 指向这个目录；这属于权重初始化，不是恢复 Stage 1 optimizer。

## 7 三路预热 A

Stage A 自动关闭 checker，不需要手动改 YAML 中的 checker 开关。以下 2000 步是起始实验设置，不是已验证的最佳训练长度。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python scripts/train_tri_input_warmup.py \
  --config configs/stage3_tri_input.yaml \
  --stage A \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_warmup_a \
  --max_steps 2000 --batch_size 1 --num_workers 0 \
  --learning_rate 0.0001 --checkpointing_steps 500 --device cuda \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/warmup_a.log 2>&1 < /dev/null &
```

```bash
tail -f /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/warmup_a.log
```

完成后目录为 `/data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_warmup_a/final`。检查损失有限、日志正常结束、`last_preview_gt.png` 无异常黑白图。预览左侧是 preview，右侧是 GT；它不是扩散超分结果。

## 8 三路预热 B

A 完成后再启动 B，使用 `--init_conditioner`，不是 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python scripts/train_tri_input_warmup.py \
  --config configs/stage3_tri_input.yaml \
  --stage B \
  --init_conditioner /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_warmup_a/final \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_warmup_b \
  --max_steps 2000 --batch_size 1 --num_workers 0 \
  --learning_rate 0.0001 --checkpointing_steps 500 --device cuda \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/warmup_b.log 2>&1 < /dev/null &
```

```bash
tail -f /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/warmup_b.log
```

完成后检查 `warmup_summary.json` 中的有效窗口、接受比例、内部误差变化和回退原因。不把这些量解释为真实性概率或 HR 精度。

确认 YAML 已有字段为：

```yaml
tri_input:
  init_conditioner_path: outputs/tri_input_synthetic_warmup_b/final
```

只更新这个嵌套字段，保留其他配置。C 开启 checker，必须加载兼容的 B 权重，直接加载 checker 关闭的 A 权重会被配置一致性检查拒绝。

## 9 正式扩散训练前两步试跑

保持 fp16，使用单独 smoke 输出目录，不覆盖正式训练目录：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  src/train_lora_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --max_train_steps 2 --train_batch_size 1 --num_workers 0 \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_smoke
```

两步指 optimizer 更新步数；梯度累积 4 时会读取多个 micro-batch。确认 loss 有限，日志显示 LoRA/Adapter 来自 Stage 1，RGB/GT 来自独立图片目录，最终保存 smoke/final。

两步也要求 train 和 val 路径有效，但不会触发当前每 500 步一次的验证。它只证明训练和保存流程能运行，不证明模型效果。原来的 float/Half 报错已通过 UNet 调用边界统一残差 dtype 修复，不需要重新预热或改成全 float32。

## 10 正式三路扩散训练 C

试跑通过后，从 Stage 1 和 B 预热权重正常初始化正式训练，不要恢复两步 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  src/train_lora_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/stage3_synthetic_train.log 2>&1 < /dev/null &
```

```bash
tail -f /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/stage3_synthetic_train.log
```

当前设置：10000 个 optimizer 更新步；batch size 1，梯度累积 4；新增模块学习率 0.0001；每 500 步保存和验证；保留最近 3 个编号 checkpoint。它沿用 Stage 1 的数据来源，但不是把 Stage 1 全部训练超参数原样复制。

主要路径：

```text
/data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/checkpoint-00500
/data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/validation/step-000500
/data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final
```

训练过程每次验证只取 5 个样本的中心裁剪，不是完整测试集。验证拼图左到右是经过原 Adapter 的 LR 放大图、SR、GT；独立推理预览使用原始 LR 放大图，二者第一栏定义不完全相同。

完整三路 artifact 包含原 LoRA/Adapter、`tri_input_conditioner.safetensors`、`tri_input_config.json`、`tri_input_bridges.safetensors`、`tri_input_bridge_config.json`、`raw_band_stats.json`、训练配置与恢复状态。推理必须使用 C 的 artifact，不能用仅有 conditioner 的 A/B artifact。

## 11 中断后的恢复方式

下面都是示例，必须先确认指定 checkpoint 实际存在且旧进程已经停止。若实际编号不同，直接修改命令中的具体编号，不要恢复到其他实验目录。早期编号可能因为保留数量限制已被删除。

同阶段 B 预热恢复，`--max_steps 2000` 表示恢复后训练到总计 2000 步：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python scripts/train_tri_input_warmup.py \
  --config configs/stage3_tri_input.yaml --stage B \
  --resume /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_warmup_b/checkpoint-00500 \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_synthetic_warmup_b \
  --max_steps 2000 --batch_size 1 --num_workers 0 \
  --learning_rate 0.0001 --checkpointing_steps 500 --device cuda \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/warmup_b_resume.log 2>&1 < /dev/null &
```

C 恢复，同结构 checkpoint 恢复新增模块、optimizer、学习率 scheduler、global step 和 RNG：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup accelerate launch \
  --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
  src/train_lora_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic \
  --resume_from_checkpoint /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/checkpoint-00500 \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/stage3_synthetic_resume.log 2>&1 < /dev/null &
```

Stage 1 → C 和 A → B 是权重初始化，不是同阶段恢复。不要把 Stage 1/final 或 A/final 填成 C 的 `resume_from_checkpoint`。

## 12 三路模型小样本推理

先同步包含精度修复的代码。下方默认 C 已训练完成；如果只有 smoke 权重，只能把 checkpoint 路径直接改为 smoke/final 做功能检查，不能评价最终效果。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --gt_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/GT_geo_rad_visual \
  --split test \
  --raw_ms_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr \
  --unmixing_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_smoke \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5 --limit 2
```

检查输出尺寸是 LR 的四倍，没有全黑、全白或明显异常。预览位于输出目录的 `previews`。不要将 `--checkpoint_path` 改为 B/final；B 没有完整扩散推理权重。

## 13 完整合成测试集推理

两张检查通过后执行全量推理。此命令没有 `--limit`、`--sample` 或 `--sample_file`，因此遍历输入目录内所有支持的图片，不受训练验证 5 张的限制。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --gt_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/GT_geo_rad_visual \
  --split test \
  --raw_ms_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr \
  --unmixing_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5 \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/tri_input_synthetic_infer.log 2>&1 < /dev/null &
```

```bash
tail -f /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/tri_input_synthetic_infer.log
```

必须看到本次日志正常结束并打印 `Saved N inference results`，再启动评估。若更换 checkpoint，使用新的明确输出目录，避免旧图残留影响评估。

## 14 完整测试集定量评估

### 14.1 模型直接输出的主要指标

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python src/evaluate.py \
  --sr_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic/sr_raw \
  --gt_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic/gt \
  --lr_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_metrics_raw \
  --device cuda --skip_lpips
```

默认加 `--skip_lpips`，避免额外初始化和潜在 AlexNet 权重下载。若已经安装 LPIPS 且缓存其权重，删除最后的 `--skip_lpips` 即可计算；初始化失败时程序会记录警告并留空 LPIPS，不能当作 0。报告必须说明是否计算 LPIPS。

### 14.2 后处理结果的单独指标

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python src/evaluate.py \
  --sr_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic/sr_projected \
  --gt_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic/gt \
  --lr_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_metrics_projected \
  --device cuda --skip_lpips
```

主要模型比较使用 `sr_raw`。`sr_projected` 是固定 alpha=0.5 的低频后处理结果，必须分开标注，不能把其收益全部归因于三路模块。

`--lr_dir` 必须是原始低分辨率 `test/LR_bicubic`，不能使用推理输出中的 `lr_bicubic`，后者已经放大四倍。当前评估器只遍历已有 SR 文件，不会自动证明完整测试集已完成；核对预检样本数、推理 `Selected N`、`Saved N` 和评估 `Evaluated N` 一致。

指标含义：PSNR、SSIM 越高越好；LPIPS、RGB MAE 越低越好；RGB bias 看是否接近 0。RGB 光谱角和 LR 重建误差只是显示 RGB 空间的辅助指标，不是 12 波段物理反射率恢复精度，不能替代目视检查。

## 15 关闭三路分支的同模型对照

这是同一个 Stage 3 checkpoint 的 RGB-only 消融，不是加载随机 backbone。无需辅助输入，其他采样参数和样本列表保持一致。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final \
  --disable_tri_input \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --gt_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/GT_geo_rad_visual \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_rgb_only \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5 \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/tri_input_rgb_only_infer.log 2>&1 < /dev/null &
```

等该任务完成后：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python src/evaluate.py \
  --sr_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_rgb_only/sr_raw \
  --gt_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_rgb_only/gt \
  --lr_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_rgb_only_metrics_raw \
  --device cuda --skip_lpips
```

比较两份 raw 指标和同名图片。固定 seed、文件选择及排序、采样步数、noise level、prompt、整图或切块方式；代码按排序后的样本序号派生随机种子，仅固定一个 seed 数字而改变样本列表仍可能改变逐图噪声。

## 16 可选原 Stage 1 基线测试

若需要单独保留原 Stage 1 的测试结果，必须同时使用 Stage 1 配置和 checkpoint。不要拿开启 tri_input 的 YAML 加旧 Stage 1 权重却不显式关闭三路。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup python src/infer_upscaler.py \
  --config configs/stage1_synthetic.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage1_synthetic/final \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --gt_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/GT_geo_rad_visual \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/stage1_rgb_baseline \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5 \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/stage1_rgb_baseline_infer.log 2>&1 < /dev/null &
```

等该任务完成后：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python src/evaluate.py \
  --sr_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/stage1_rgb_baseline/sr_raw \
  --gt_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/stage1_rgb_baseline/gt \
  --lr_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/stage1_rgb_baseline_metrics_raw \
  --device cuda --skip_lpips
```

## 17 可选真实 RGB 泛化测试

这是在真实 `test/LR` 上测试用合成 RGB 训练的模型，不是 Stage 2 训练，也不是与合成数据同分布的测试。先确认真实 LR、GT 与 test 辅助数据仍满足同名、四倍尺寸和空间对应约定。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR \
  --gt_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/GT_geo_rad_visual \
  --split test \
  --raw_ms_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr \
  --unmixing_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_real \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5 \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/tri_input_real_infer.log 2>&1 < /dev/null &
```

等该任务完成后：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python src/evaluate.py \
  --sr_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_real/sr_raw \
  --gt_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_real/gt \
  --lr_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_real_metrics_raw \
  --device cuda --skip_lpips
```

## 18 可选无 GT 和切块推理

### 18.1 无 GT 推理

去掉 GT 参数即可，下面用同一个已知测试输入目录演示，不要求读取任何 GT：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --split test \
  --raw_ms_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr \
  --unmixing_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_no_gt \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5
```

无 GT 时不能使用现有全参考 `evaluate.py` 计算 PSNR/SSIM/LPIPS。它仍会创建 `gt` 目录，但不写入 GT 图片；预览只有三个面板。

### 18.2 切块推理

显存不足时使用，三路数据按相同 LR 坐标切块，输出使用原 Hann 融合。小于 tile size 的图片显式转整图推理，不放大输入。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 nohup python src/infer_upscaler.py \
  --config configs/stage3_tri_input.yaml \
  --checkpoint_path /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/stage3_tri_input_synthetic/final \
  --input_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --gt_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/GT_geo_rad_visual \
  --split test \
  --raw_ms_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr \
  --unmixing_dir /data/zhengay/EDiffSR-main/data/new_star/test/lr_unmixing \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_tiled \
  --num_inference_steps 40 --noise_level 10 --guidance_scale 1.0 \
  --prompt_mode fixed --seed 42 --mixed_precision fp16 \
  --low_freq_projection_alpha 0.5 \
  --tiled --tile_size 128 --tile_overlap 32 \
  > /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/tri_input_logs/tri_input_synthetic_tiled_infer.log 2>&1 < /dev/null &
```

等该任务完成后：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python src/evaluate.py \
  --sr_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_tiled/sr_raw \
  --gt_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_tiled/gt \
  --lr_dir /data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/test/LR_bicubic \
  --output_dir /data/zhengay/StableSR/worldstrat_sd_upscaler/outputs/full_test/tri_input_synthetic_tiled_metrics_raw \
  --device cuda --skip_lpips
```

切块与整图不是严格相同的采样过程，比较模块作用时两组必须使用同一种方式。

## 19 输出文件阅读顺序

| 输出 | 含义 |
|---|---|
| `sr_raw` | 模型直接输出，主要质量评估对象 |
| `sr_projected` | 低频投影后的后处理结果 |
| `lr_bicubic` | 本次输入 RGB 的 bicubic 四倍放大图，不是原始低分辨率目录 |
| `gt` | 若提供 GT，复制同名 GT 用于对比 |
| `previews` | 独立推理拼图：原始 LR 放大、SR raw、SR projected、GT |
| `per_image_metrics.csv` | 每张图的指标 |
| `summary_metrics.csv` | 每项指标的均值、中位数、标准差、5/95 百分位和有效样本数 |
| `summary_metrics.json` | 同一汇总的 JSON 形式 |

不要混淆三种预览：A/B 是 preview 与 GT 两栏；训练验证是适配后 LR、SR、GT 三栏；独立推理是原始 LR、raw、projected、GT 四栏。`GT_geo_rad_visual` 已在拉伸显示 RGB 空间处理，所有 RGB 误差只在该空间解释。

先看 raw 的同名拼图及最差样本，再看汇总指标，最后单独讨论 projected 后处理。checker 的 support、scores、内部改善和接受比例只解释候选选择行为，不代表真实 HR 精度或校准置信度。

## 20 常见启动错误和执行边界

| 情况 | 处理 |
|---|---|
| `unrecognized arguments: --split test` | 三路推理使用 `src/infer_upscaler.py`，不是 Cas 专用 `infer_stage1_latent_phi.py` |
| 找不到 `sr_raw` | 先确认推理成功结束和输出路径一致，不要在后台推理刚启动时评估 |
| 缺少三路权重 | 推理应使用 C checkpoint，不是 A/B 预热目录，也不是未声明关闭分支的 Stage 1 权重 |
| `raw_stats_path` 缺失或统计不匹配 | 先生成并正确配置 train-only 统计，推理使用模型保存的那份 |
| 原始 `Input type float and bias Half` | 同步桥接与主训练的 dtype 修复；无需为此重新预热 |
| 显存不足 | 停止同卡其他任务；推理可用独立 tiled 实验；不要默默改变正式比较设置 |
| checkpoint 编号不存在 | 按实际保留目录指定；只保留最近三个时早期编号可能已删除 |
| `nohup: ignoring input` | 正常提示，不是训练失败；继续看后续日志 |

本次文档核验基于本地 commit `ff2560a04189cf75873fa5a41ec399689cfc2fa8`：35 个 Bash 代码块通过 `bash -n` 语法检查，其中 27 条脚本命令通过从当前源码提取的真实 argparse 参数解析；检查时没有执行训练、推理或下载。仅修改本手册及 README 入口，没有修改模型、训练或推理逻辑。

所有命令已按当前仓库接口整理，但本手册不代表服务器任务已运行或模型效果已验证。数据目录、完整权重、实际 GPU 显存及完整测试覆盖仍需在服务器执行确认。不得将 2 步 smoke 或仅 5 张训练验证说成完成全测试集评估。
