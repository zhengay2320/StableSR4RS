# StableSR4RS：三路输入与局部光谱检查基础开发任务

请在当前真实仓库中完成代码修改、测试和文档，不要只给设计建议、独立示例或另建一个没有接入训练的项目。按阶段推进，每阶段给出实际修改与测试结果。不要修改、重算、移动或覆盖任何原始数据和离线解混文件；不要启动完整训练或自动下载大型模型。

## 1. 开始前确认真实工程，不按仓库名称猜架构

仓库：https://github.com/zhengay2320/StableSR4RS.git

本任务编写时核查的 main 提交为 bbe54a287920da88784c5e08a93cc94b06ef3a02。此 SHA 只用于说明审阅基线，不要求回退到它；以当前本地代码为准，记录实际 commit、已有未提交修改及相关 AGENTS.md，保护已有工作，不做 reset --hard、清理用户文件或无关大规模重构。

主要修改对象是 worldstrat_sd_upscaler/，不是 sisr4rs-main/。当前该子项目基于 Diffusers StableDiffusionUpscalePipeline / stabilityai/stable-diffusion-x4-upscaler，README 声明绑定 Diffusers v0.39.0。它不是原版 StableSR 仓库，不要套用不存在的 time-aware encoder、原版 SFT 或 SDXL 接口。

先阅读：
- worldstrat_sd_upscaler/README.md、requirements.txt、pyproject.toml、scripts/setup_env.sh；
- src/dataset.py、condition_adapter.py、train_lora_upscaler.py；
- src/infer_upscaler.py、evaluate.py、utils.py；
- src/diffusion_prediction.py、latent_phi.py、infer_stage1_latent_phi.py、rgb_auxiliary_loss.py；
- configs/stage2_cross_sensor.yaml，以及当前实际使用的其他配置；
- tests/ 中现有测试。

另外阅读两个参考文件（在当前工作目录或用户提供的附件位置查找）：
1. 光谱解混模块_离线处理与融合接口说明.md；
2. tri_input_explained_v1.zip，重点阅读 tri_input_module.py、README_使用说明.md、test_prototype.py。

附件是原型参考，不是已接入仓库的正式代码。可安全检查后迁移、拆分和修正，不能盲目复制后宣称完成。若附件未出现在 Codex 工作环境，不要假装读过；记录这一点，按本任务完整规格继续实现。也可只读参考仓库 spectral_unmixing_tensor_batch/ 中的格式与数值约定，但禁止在训练端导入并执行其解混流程。

先输出简短审计摘要：实际入口、batch 字段、RGB 数值范围、模型输入/latent/特征形状、checkpoint 文件、合成回放机制、推理与验证路径，以及计划修改文件。完成审计后继续开发，不停留在计划。

## 2. 本次范围：实现已经讨论的离散候选版本

最终任务保持 RGB 四倍超分，输入三部分：
R = 现有模型使用的处理后 RGB；
M = 同场景同网格的原始 12 波段 Sentinel-2；
T = 离线五类丰度 F + 对应五类模型敏感性 U。
输出仍为高分辨率 RGB，不增加高分辨率非 RGB 输出任务。

新增逻辑：
三路编码 -> 初始软结构 Q0 -> 小范围候选光谱检查 -> 修正结构 Q* -> 几何/光谱条件 -> 现有扩散模型。

光谱检查的原则是：每个候选布局分别重新拟合少量局部共享光谱系数，再比较剩余误差；先允许光谱解释改变，不能看到光谱残差就立即移动结构。

本次不实现连续 Jacobian/Schur 补、完整后验采样或第二个大型扩散模型。离散选择不声称端到端可微，光谱内部评分不声称是真实 HR 精度或校准概率。

保留 LoRA、原 ConditionAdapter、原 scheduler/预测参数化、原 RGB 辅助损失和原数据路径。新增分支默认不影响旧配置；第一版新配置关闭 phi_enabled，不删除旧 phi/Cas 功能。若三路分支与 phi 尚未联合支持，组合开启时明确报错，不能静默忽略其中之一。

## 3. 路径与样本配对：这是硬约束

用户明确提供：
- train 原始12波段目录：/data/zhengay/EDiffSR-main/data/new_star/train/lr
- train 离线解混目录：/data/zhengay/EDiffSR-main/data/new_star/test/train_unmixing

第二个路径虽然在 test/ 下，但用户明确将它与 train/lr 配对。必须原样保存，不得改成 train/lr_unmixing，不得仅根据父目录名称判断它属于测试集。

用户已完成 train、val、test 的离线解混，但没有给出 val/test 的确切目录。若当前服务器可访问，通过只读目录检查核实并记录；否则分别配置为 null/必填项，列出运行相应 split 需要补齐的键。不得擅自推断为 test/val_unmixing、val/lr_unmixing 等路径；不得把 train 解混目录复用于 val/test。

处理后 RGB 和 GT 沿用当前 data_root、LR、GT/GT_geo_rad_visual 等实际配置，不能把 new_star/train/lr 直接当作 PIL RGB 目录。仓库现有配置中的处理后 RGB 根目录与 new_star 原始多光谱目录是两套路径。

为每个 split 单独配置 raw_ms_dir 和 unmixing_dir。RGB-GT 维持旧的配对规则；新增 raw TIFF 与 NPY 优先通过 sample_id/唯一 stem 对应，允许 RGB 为 .png、raw 为 .tiff、prior 为 .npy。
- 不使用目录排序位置配对。
- 同 stem 对应多个 TIFF、漏文件、重复样本必须报错并指出样本名与路径。
- 文件名确实经过重命名时，支持显式 manifest，不编造模糊匹配规则。
- 验证三路地理范围与网格是否一致，尺寸相等不是地理对齐证明；缺地理元数据时依靠明确的数据制作约定并记录，禁止自动缩放后假定已配准。
- 不移动磁盘文件来迎合配置；只在 outputs/ 写索引、统计和报告。

## 4. 数据契约与预处理

建议保持旧 batch 字段，并在三路模式添加：
- lr：原有处理后 RGB，[B,3,H,W]，保持原来的 [-1,1] 约定；
- gt：原有 RGB 标签，[B,3,4H,4W]，仅训练/评价使用；
- raw_ms：[B,12,H,W]，float32，使用明确数值单位，不进行显示拉伸；
- unmixing：[B,10,H,W]，float32；
- raw_valid：有效观测掩膜，可选逐波段形式；
- aux_present：用于显式消融/回放控制，不用于掩盖文件缺失；
- 原有 sample_id、filename、source_type、prompt，并保留 crop/flip/rotation 元数据用于诊断。

读取 raw TIFF 使用能够保留12个波段和元数据的库，例如 rasterio，绝不能 PIL.convert('RGB')。若新增依赖，在项目依赖中明确记录，保留旧 RGB-only 环境的可用性，尽可能延迟导入。

原始波段顺序必须显式指定并验证：恰好12个不重复的 L2A 波段、RGB索引、B1/B9索引。标准顺序只能作为待核实示例，不能根据通道数默认断言。核对当前离线预处理和实际 TIFF 的 scale、offset、reflectance/DN 约定；不能无条件除以10000、不能重复应用偏移。无法确认时提供明确配置和启动检查，不猜测。

原始数值与网络标准化分开：
- 按确认的单位转换到 raw_ms；
- 使用 train 集有效像元计算12通道 mean/std，保存 JSON、通道顺序、数据来源、转换参数和版本摘要；
- val/test/inference 只读训练统计，不能参与估计，也不做每图 min-max；
- 提供统计脚本，不能用全0均值/全1标准差作为真实实验的静默默认；
- 不重复标准化。光谱检查可使用与原型一致的逐波段标准化观测，但必须在文档中说明分数所在空间。

离线特征严格校验：np.load(..., allow_pickle=False)，float32 [10,H,W]、全部有限、范围[0,1]，并与对应原始 TIFF 网格一致。
F=T[:5]，U=T[5:]；类别顺序 vegetation,water,bare,snow,building。
不能对T整体Softmax，不能强制F和为1；F=0,U=1是不可直接信任的占位/高分歧，不是真零覆盖。U不是概率、方差、RMSE；低U也可能错误。

不重新生成或改变F/U。无效影像的处理：数据错误先报错；明确nodata/padding使用掩膜、进入网络前有限值填充，回归拟合和评分均剔除它们。不能把U当云影质量掩膜，也不能仅因零反射率就判为nodata。无有效观测时几何检查返回“不动”。没有语义云掩膜时，明确需要已筛选清晰数据，不宣称已检测云。

所有空间增强共享一次采样：同一LR裁剪框作用于R/M/T，GT使用4倍框；同步翻转旋转，禁止各分支独立随机增强。F/U不接受RGB颜色扰动。保持旧模式的行为和随机数使用尽量不变。

## 5. 必须处理仓库现有合成回放

现有 stage2 有 synthetic_replay_probability，会将真实LR替换成同名 LR_bicubic。合成LR不等于产生当前原始光谱/FU的真实观测。

新增三路真实训练配置默认：train_lr_subdir: LR，synthetic_replay_probability: 0.0。
不得把原来0.4等回放值不加检查继承进新实验。

若之后显式启用回放，只支持清楚的政策：
- 默认检测冲突并报错；或
- 用户显式选择 synthetic_aux_policy=disable 时，对该合成样本关闭整个新增辅助分支及相关损失、日志标记，走旧RGB路径。
不能对合成RGB静默配真实M/F/U，不能给回放假造多光谱，也不能仅关闭解混而继续错误使用raw。
旧RGB-only合成训练配置不改变。

## 6. 模型实现：迁移并完善参考原型

### A. TriInputConditioner

参考初始通道数：RGB 3->32；raw 12->48；prior concat(F*w,U,使用标记) 15->32；融合112->64；两次PixelShuffle；8通道softmax得到Q0，空间为4H×4W。
前5个内部响应用解混作弱提示，另3个自由响应，不声称8通道是分割真值。
- raw所有12波段参与上下文；
- 全未知prior只关闭prior信息，不关闭raw；
- 输入T detach、只读；
- 不依赖当前扩散z0或HR标签生成条件；
- 条件在每样本/每tile的一次采样之前计算一次，不每个denoising timestep重算几何。

### B. LocalSpectralChecker：五候选近似

默认参数：scale=4，common_factor=2，window=8，stride=4，max_shift_hr=1.0，ridge=1e-4，movement_penalty=1e-4，min_relative_gain=0.02，min_relative_gap=0.005，absolute_gain_floor=1e-7。它们是可配置原型起点，不是已验证最优值。

1. 候选为不动、左、右、上、下；位移单位是目标网格像元，不是LR像元。warp只作用于Q0，不改变R/M/T或GT。
2. 在相同公共检查网格聚合Q和原始十个地表波段，排除B1/B9参与几何回归，但不排除其上下文编码。
3. 每个窗口、每个候选分别用局部共享线性系数解释光谱：
   A:[N,K]，Y:[N,10]；
   E=solve(A.T@A/N+ridge*I, A.T@Y/N)。
   不能每个高分像元独立拟合系数，不能把它叫重新解混/FCLS或真实端元恢复。
4. 两组2×2空间块交错划分，互相拟合与评分，再平均；有效掩膜作用于拟合和评分，样本不足、矩阵病态或数值异常时带诊断返回不动。它是内部检查，不是独立验证、概率或严格物理似然。
5. 加位移惩罚；仅最佳候选显著优于不动且与次佳有足够差距时接受，否则不动；平局、无信息都优先不动。
6. 重叠窗口融合有限位移，再warp一次Q0；不能平均候选RGB。平滑后重新评分，不改善则该样本回退。只声称样本平均内部目标不恶化，不声称逐像元真实性保证。
7. V1候选评分和选择放在no_grad；最终warp对Q0保留梯度。小型线性求解显式关闭AMP，使用float32；必要时受控提高精度或阻尼，并记录，不静默吞异常。

完善附件原型边界：支持非方形和非整除尺寸、末端窗口、有效padding；checker关闭时必须立即绕过窗口形状限制。通过padding或末端窗口覆盖，不对奇数尺寸静默resize。图像过小可明确拒绝或记录后不做几何检查；不能产生零覆盖除法或NaN。
平均池化只是共同支持尺度近似，观察算子独立封装，注释不得宣称真实PSF反演。

### C. Geometry/context 条件

输出至少包含：q0、q_star、flow_hr、support、scores、geometry金字塔、context_lr、可选preview。
64×64例子可对应：
geometry = [32×256×256,64×128×128,96×64×64,128×32×32]，context_lr=64×64×64。
这些是条件模块内部尺寸，不是宿主VAE尺度。
几何编码使用Q*、Q*-Q0、位移、支持提示；context来自raw编码。context必须在无解混时仍有效。

preview仅用于新增模块预热；不要把preview当最终扩散结果。预热可用现有HR RGB，不要求新增HR语义标签。

## 7. 真正接入当前 Diffusers x4 模型，不只保存一个独立模块

审计真实UNet配置和运行时特征尺寸后，采用零初始化的特征残差连接。至少接入一个空间特征位置与一个额外尺度，或提出并实现经测试等价的官方适配接口。优先明确的wrapper/正式forward参数；不要仅将条件图拼到文本token或仅返回而不使用。

特别约束：
- 保持现有UNet输入契约。当前训练拼接noisy_latents与noisy_low；实际通常为4+3通道，以运行时配置为准。不得将第一层直接改成25/29通道后破坏旧权重。
- 不能假定此x4模型的VAE一定是8倍压缩。动态读取pipe.vae_scale_factor和实际latent形状；geometry与特征位置通过显式投影匹配。
- 原ConditionAdapter与low_res_scheduler加噪路径保持。新模块的R取当前预处理RGB条件，在low_res_scheduler随机加噪前使用；训练/验证/推理一致，不把latent噪声图作为其输入。
- 参数由明确的nn.Module拥有、注册，并进入optimizer、Accelerator/DDP、梯度裁剪与保存；不能把参数藏在未注册闭包hook中。
- 冻结主干参数不等于全程no_grad；最终扩散损失必须能经主干传回新增条件分支。
- 如使用hook，必须有严格作用域、try/finally清理、单实例避免重复注册，并验证gradient_checkpointing的backward重计算仍使用正确条件；不能跨batch/tile留状态。未经测试的持久全局monkey patch不接受。
- 同一条件接入机制必须用于train forward、run_validation、普通推理和分块推理，不能只有训练有效。
- CFG时按当前pipeline真实样本重复顺序扩展条件，匹配num_images_per_prompt和[negative,positive]顺序；遵循本图像条件pipeline对原图条件的复制策略，不凭空把一半raw清零。测试B>1、CFG>1及多图请求。
- 条件预计算不能额外消耗pipeline采样使用的随机数流。关闭全部新增功能时必须回到旧路径；不要求raw/NPY文件存在；同权重、同seed、同scheduler应复现旧结果。
- 零初始化桥接下与旧模型数值对齐；第一步上游梯度可能因零桥接为零，验收在至少两个更新或非零桥接情形下检查，不伪造“所有分支首步有梯度”。

第一版新配置冻结已有LoRA、原ConditionAdapter、VAE、text encoder和UNet主参数，只训练新增模块/桥接。允许配置解冻LoRA/原Adapter，但默认关闭。当前get_trainable_parameters对无可训练参数会报错，需相应调整新路径的optimizer构建、DDP包装和裁剪，不能为了满足旧函数而偷偷解冻旧参数。

## 8. 配置、脚本与训练阶段

不要改写原实验YAML；新增例如 configs/stage3_tri_input.yaml。精确命名可遵循实际工程，但文档/代码/脚本必须一致。
建议新增结构：

tri_input:
  enabled: true
  train_existing_lora: false
  train_existing_condition_adapter: false
  synthetic_aux_policy: error
  data:
    train:
      raw_ms_dir: /data/zhengay/EDiffSR-main/data/new_star/train/lr
      unmixing_dir: /data/zhengay/EDiffSR-main/data/new_star/test/train_unmixing
    val:
      raw_ms_dir: null
      unmixing_dir: null
    test:
      raw_ms_dir: null
      unmixing_dir: null
  band_names: null
  raw_value_conversion: null
  raw_stats_path: null
  components: 8
  checker:
    enabled: true
    common_factor: 2
    window: 8
    stride: 4
    max_shift_hr: 1.0
    ridge: 0.0001
    movement_penalty: 0.0001
    min_relative_gain: 0.02
    min_relative_gap: 0.005
    absolute_gain_floor: 0.0000001

synthetic_replay_probability: 0.0
phi_enabled: false

null意味着必须确认，不是让代码自动猜。启动前输出准确的缺失配置及对应检查命令。可以先做不依赖真实路径的CPU测试与fixture smoke；不要因此停下所有开发。

实现并提供真实参数的命令/脚本：
A. 三路数据预检、匹配manifest和数值摘要；
B. 仅train原始波段统计；
C. conditioner preview预热（不要求下载扩散模型），先checker关闭再可选开启；
D. 从现有已训练RGB checkpoint初始化三路扩散训练；
E. 独立验证、无GT推理和tiled推理；
F. CPU smoke与可选真实GPU smoke。

现有init_lora_path/init_adapter_path沿用或显式配置，不自动挑选“最新checkpoint”，不虚构已经训练好的文件。必要模型/统计文件缺失时，真实训练明确失败并给出路径信息，不能使用随机权重假装完成联调。

preview与HR的range沿用旧RGB处理；不新加未经验证的RGB->SWIR“物理损失”。候选搜索no_grad，其分数只做诊断，不宣称直接优化了它。最终联合训练仍使用现有扩散目标，可选preview辅助权重写入配置并说明。

## 9. 验证、推理与tiled全链路

扩展现有infer_upscaler的split/aux根路径或manifest参数，保留原有命令兼容。处理input_dir时不能从目录名称或checkpoint猜解混split。

1. 不给gt_dir也能正常三路推理；HR只用于训练目标和可选评价。
2. 普通推理按同样stem读取raw和NPY；模型只加载一次，条件每样本算一次。
3. 分块时R/M/T使用完全相同LR坐标、相同外扩/裁边；每tile一次计算条件，不在每个denoising步重算。
4. 原模型要求的padding与checker要求的padding联合设计。RGB可保留既有edge pad；raw填充伴随invalid mask；F=0,U=1填充。检查时绝不把padding当真实观测。输出按4倍原始尺寸裁回。
5. 奇数和长方形图像必须处理；小于tile_size可明确走whole-image，不能暗中放大。
6. Hann blending与seed规则沿用原实现，边缘权重不能为零；几何与上下文cache不能串到下一tile/图像。
7. 保留sr_raw与sr_projected分开保存的既有逻辑。新模块不偷偷改变projection alpha，不移动GT找最优评分。
8. 验证和独立推理调用共用的条件准备/采样辅助函数，避免train正确而validation漏条件。

CFG、num_images_per_prompt、dtype、device、sample筛选、padding都纳入测试。需要时在新checkpoint启用但条件不可获得的情况下fail-fast，而不是默认退化成RGB并仍称为新模型结果。

## 10. 保存、加载与恢复

保留现有pytorch_lora_weights.safetensors、condition_adapter.safetensors、training_config.yaml和已有artifact加载逻辑。
新增独立模块权重与结构元数据，例如：
- tri_input_conditioner.safetensors
- tri_input_bridges.safetensors（也可以在注册结构下合并，保证完整）
- tri_input_config.json
- raw_band_stats.json

记录版本、实际波段顺序、数值转换、归一化、checker配置、宿主hook/注入位置及通道、训练阶段和依赖的RGB checkpoint。
旧artifact不含tri_input配置时按旧RGB模式加载；用户显式要求tri_input但文件缺失时严格报错。
新checkpoint要能无损恢复新模块、optimizer、scheduler、global_step和相关RNG状态；旧RGB初始化与同结构训练恢复分别处理，不把旧optimizer强行加载到新参数组。
支持map_location与dtype转换，保存后重新加载需通过同输入输出一致性测试。不能泛用strict=False吞掉任意缺键；只允许明确白名单的向后兼容差异并报告。
三路完全关闭的消融必须恢复旧路径，不要求aux数据加载，也不触发新分支随机数消耗。

## 11. 建议文件组织，按实际工程调整而不是另起项目

可新增：
- src/tri_input_conditioner.py
- src/local_spectral_checker.py
- src/tri_input_bridge.py
- src/tri_input_data.py（或在dataset.py适度扩展）
- scripts/validate_tri_input_data.py
- scripts/compute_raw_band_stats.py
- scripts/train_tri_input_warmup.py
- configs/stage3_tri_input.yaml
- docs/tri_input_integration.md
- tests/test_tri_input_*.py

必须实际接通dataset.py、train_lora_upscaler.py的build_datasets/训练/验证/保存，以及infer_upscaler.py的whole-image/tiled路径。可抽取公共helper，但避免复制整份训练脚本产生两套不一致逻辑。
不修改sisr4rs-main或离线解混算法，不直接改third_party的Diffusers源代码来实现业务功能。保持当前版本与依赖体系，不无说明升级Diffusers或替换PyTorch。

## 12. 验收：运行后报告，禁止只写“理论可行”

至少提供以下不下载大模型的CPU测试：
- stem正确配对、扩展名不同、重复stem、缺raw/NPY报错、split独立；
- 离线dtype/shape/range/NaN、未知编码，F不被重归一化；
- 带坐标编码的三路同步裁剪/翻转/旋转，GT4倍对应；
- 原始波段顺序、数值转换、统计只用train，统计随checkpoint复用；
- 合成回放禁止错误配真实光谱；旧RGB-only回放不受影响；
- 全未知prior仍保留raw分支，显式关闭新增分支回到RGB基线；
- 三路尺寸、checker开关、非方形/奇数尺寸、不足窗口、masked padding；
- 五候选位移方向/单位；纯光谱变化、已知可辨识结构偏移、无信息与平局；
- float32求解、有限数值、有效位置不足时可追踪回退；
- no_grad内部选择与最终Q0梯度传递的边界；
- 新参数注册、optimizer包含、冻结旧模块不被更新，多步后各新分支梯度；
- bridge关闭/零初始化等价、连续两样本不同条件不串缓存；
- 实际扩散调用链的轻量集成测试：用本地初始化的小型兼容UNet或真实结构的mock，不只测conditioner；
- CFG、batch>1、num_images_per_prompt的排列和形状；
- gradient checkpointing重计算条件和梯度；
- 保存/重载、旧checkpoint兼容、新checkpoint缺权重报错；
- 无GT whole-image与tiled路径的原始4倍尺寸和辅助输入同步。

用CPU fixture验证端到端文件读取到新条件进入UNet的闭环。真实环境可用时再做1-2个真实样本、至少2次optimizer更新、保存/重载、一次无GT推理；按显式开关触发，绝不自动完整训练。

执行现有相关tests和新增tests，记录实际命令/退出码/通过数。依赖、模型、GPU或用户/data路径不存在时，明确区分“通过”“跳过”“阻塞”“未验证”，提供在用户服务器上执行的准确命令。不能伪造PSNR提升、真实数据测试或预处理统计。

日志至少含checker启用状态、有效窗口数、接受位移比例、平均位移、内部改善与回退数、三路有效比例、RGB/扩散/preview损失；support与score始终标记为内部启发式量，不叫真实置信概率。

最终交付：
1. 实际修改文件和各自作用；
2. 三路数据从读取到train/validation/inference的真实函数调用链；
3. init/resume/禁用/缺失/回放的行为说明；
4. 完整训练、验证、无GT推理、tiled推理命令；
5. 单测及集成测试结果；
6. 所有待补路径、band map、数值参数或真实环境验证项；
7. 当前方法仍是离散候选原型、共同尺度是近似、内部光谱分数不证明HR真实性等边界。

完成标准是现有仓库具备可配置、可保存、可恢复、训练和推理一致的三路方案，而不是新增一个无法被现有入口调用的.py文件。
