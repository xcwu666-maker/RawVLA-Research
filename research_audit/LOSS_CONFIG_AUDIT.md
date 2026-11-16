# Audit B — Loss 与训练配置

审计日期：2026-11-16（Asia/Shanghai）。执行者：独立 Agent B。只读审计；未导入大模型、未做前向/反向、未训练、未下载权重或数据。

- 根版本：`ba9c56b04da2c4931b70d8844710f5928d43775b`；StarVLA：`57d508cb180d0510ac9060358009c0f6ffaac420`。
- 下文 `starVLA/...` 路径均指根仓库锁定的子模块，不能用上游 StarVLA 最新版代替。
- `Confirmed`：论文原文、锁定源码或本次静态命令直接支持；`Partially Confirmed`：静态实现支持，真实训练未运行；`Unknown`：缺少作者运行配置/日志等证据。
- 论文来源：[arXiv v1 §3.4](https://arxiv.org/html/2609.37530v1#S3.SS4)、[Appendix D.1](https://arxiv.org/html/2609.37530v1#A4.SS1)、[D.3 / Table 6](https://arxiv.org/html/2609.37530v1#A4.SS3)。公式用锚点定位，代码用锁定版本行号定位。

## 1. 首要结论

**Confirmed：论文的 `λ_b=0.001` 与当前 LIBERO 配置的 `exposure_loss_weight=0.01` 指向同一类 brightness 辅助损失的外部系数，但数值不等效。** 当前实现的 two-sided L1 中未发现抵消十倍差异的 `0.1` 因子。不是因为 burst 六帧、recurrent 六步或梯度累积而自然等效。审计能确认当前发布配方与论文文字存在差异；缺少作者实际运行产物，不能据此断言历史实验有 Bug，也不能确定应改代码还是修正文稿。

证据：`configs/base/rawvla.yaml:34–37,49`；`starVLA/starVLA/model/modules/raw_frontend/registry.py::_RAWVLAFrontend.forward:368–389`；`starVLA/starVLA/training/train_starvla.py::VLATrainer._train_step:640–661,853`。代码逐次将原始辅助标量乘对应配置系数，再对总损失 backward。

**Confirmed：RoboTwin 当前配方还采用 paired RGB chroma supervision，不能直接视为论文宣称的无配对重建监督训练。** `configs/base/rawvla_robotwin.yaml:8–9,39–45,49–59` 给出 RGB 根目录、曝光目标 0.6、曝光权重 0.03、chroma prior 权重 0、paired chroma supervision 权重 0.3，并只训练 1,000 步。论文 D.1/D.3 写的是曝光目标 0.48、两项辅助权重均 0.001、2,000 步、paired default-light RGB 只作无梯度 checkpoint selection。两者均保留 frozen VLA action loss；paired chroma loss 是额外训练项，不是选择 checkpoint 的指标。

## 2. 公式 (10)、(11)、(12) 与代码对应

| 项目 | 论文定义 | 当前对应实现 | 证据等级 |
| --- | --- | --- | --- |
| (10) `L_action` | 采用 frozen VLA 原生动作目标，梯度通过处理后图像更新 ISP | Qwen3-OFT mean L1；PI0/PI05 mean flow-matching MSE；模型返回 action_loss | Confirmed（静态定义） |
| (11) `L_bright` | 每张图像均值与 0.48 的绝对差，再对 batch 平均 | `_RAWVLAFrontend.forward` 的 two_sided_l1：`abs(per_image_mean-target)`，再 mean；LIBERO target=0.48 | Confirmed |
| (11) `L_chroma` | 输出 log-chroma descriptor 到 unpaired RGB envelope 的 smooth-L1 距离 | 默认 `output_envelope` / `mean_log`：有效像素 log(R/G)、log(B/G)均值；envelope 扩展 margin=0.1；越界量除原 envelope width，再 smooth-L1(beta=.25)，默认 mean reduction | Confirmed（结构对应；原始 envelope 标定样本来源未核实） |
| (12) `λ_a` | 随 backbone 改变 | `trainer.action_loss_weight` | Confirmed |
| (12) `λ_b` | 0.001 | trainer 直接读取 `framework.raw_frontend.exposure_loss_weight` | Confirmed |
| (12) `λ_chroma` | 0.001 | trainer 直接读取 `framework.raw_frontend.chroma_prior_weight` | Confirmed |

原生 action loss 证据：

- `starVLA/starVLA/model/framework/VLM4A/QwenOFT.py::__init__:146` 使用 `nn.L1Loss()`；`forward:214–234` 对预测动作和 target 计算并返回辅助 losses。
- `starVLA/starVLA/model/framework/VLM4A/PI0.py::forward_from_observation:474–494` 生成 flow targets，使用 `F.mse_loss(...,reduction="none").mean()`；`forward:499–514` 合入 auxiliary losses。
- `starVLA/starVLA/model/framework/VLM4A/PI05.py::PI05:50–58` 继承 PI0，替换 action head，无额外总 loss 乘数。
- `starVLA/starVLA/model/modules/raw_frontend/registry.py::get_raw_frontend_aux_losses:665–671` 返回 `last_aux_losses`，没有重标定。

辅助 loss 的精确实现：

- 曝光：`registry.py::_RAWVLAFrontend.__init__:131–157` 默认 `two_sided_l1`、STE clipping、domain multiplier 为 1；`forward:339–389` 只对最终 recurrent output 构造目标，RGB 三通道及 H/W 求均值，然后 batch/view 平均。默认配置没有自定义 `exposure_loss_multiplier_by_lighting_domain`；`extremelow` 的默认 multiplier 也是 1。
- Chroma：`registry.py::_chroma_descriptor:275–285`，mask 为 denoised RAW 均值 >.01 且最大值 <.995；输出 RGB 先下限 clamp 至 1/255；计算两维 log ratios。`forward:500–514` 使用 `width=max(upper-lower,1e-4)`，扩展上下界各 `margin*width`；`violation=relu(lower-descriptor)+relu(descriptor-upper)`；`smooth_l1(violation/width,0,beta=.25)`。配置 envelope 见 `configs/base/rawvla.yaml:44–52`。
- 注意不能误把 `baselines/rawvla/rawvla.py::exposure_prior_loss:558–573` 的单边四次方函数当当前训练目标。当前 wrapper 自己根据 `exposure_loss_type` 分支计算；`two_sided_l1` 走 `registry.py:387–389`，不调用该四次方函数。`registry.py:264` 虽保存函数引用，不等于前向使用。
- `trainer.loss_scale.vla:1.0` 虽出现在 RoboTwin YAML，当前 `train_starvla.py::_train_step` 调的是 `self.model.forward(batch_vla)`，不是 `BaseFramework.compute_loss(tag,loss_scale=...)`；所以该键不能用来解释额外 0.1。证据 `train_starvla.py:640–661`；另一个可选接口才在 `starVLA/starVLA/model/framework/base_framework.py:181–207` 读取 loss_scale。**Confirmed（此入口）**。

## 3. 梯度路由与配对 RGB 项

**Confirmed（静态实现），实际梯度数值未测。**

- `baselines/rawvla/rawvla.py::RAWVLA.exposure_objective_rgb:941–963`：split 模式曝光分支 detach 白平衡与 CCM；保留 exposure、tone、denoised RAW 的梯度。
- `RAWVLA.chroma_objective_rgb:965–984`：detach denoised RAW、exposure、tone，保留 WB/CCM 梯度。
- 部署 action 路径仍使用普通 output.rgb，允许原生动作目标通过完整 ISP 回传；辅助分支使用 STE clamp（`configs/base/rawvla.yaml:37`，上述两个函数）。
- `registry.py::_RAWVLAFrontend.forward:332–337` 循环六个 endpoints，将 state/theta 传入下一步，没有 detach；仅最终 output 计算辅助 losses。六次 recurrent 调用不是把曝光损失额外乘六。
- `train_starvla.py::prepare_training:162–205` 冻结指定 backbone，并验证只允许 raw_frontend 参数可训练；`_enforce_frozen_modules_eval:138–147` 只设 eval，不使用 no_grad 包住训练前向。

RoboTwin paired supervision 的证据链：

1. `configs/base/rawvla_robotwin.yaml:8–9` 设置 RAW/RGB 对应根目录。
2. `starVLA/starVLA/dataloader/rawvla_npz_datasets.py::_load_rgb_episode:137–149` 按同相对路径载入 RGB；文件缺失直接报错。
3. 同文件 `__getitem__:267–269,290–293` 当前帧 RGB /255 形成 `rawvla_chroma_target_float32`。
4. `PI0.py::_apply_raw_frontend:278–294` 将其传入 frontend；PI05 继承该实现。
5. `registry.py::forward:399–464` detach target，计算预测与 target 的三通道 normalized chromaticity `RGB/sum(RGB)` smooth-L1(beta=.05)，目标亮度 mask 范围 [.02,.98]；对有效像素均值，再 batch 均值；domain/highlight multiplier 默认 1。
6. `train_starvla.py:655–661` 将它乘 `.3` 加入总损失。预测图像使用 `output.rgb`，不是限制 WB/CCM 的 chroma_objective_rgb，因此此额外项的梯度路径也不等于论文的弱 envelope prior。

因此，若 paired RGB 可用，当前 RoboTwin 无 CLI override 的公式为：

```text
PI0 : 0.02 * L_flow + 0.03 * mean(abs(mean(Y_aux)-0.6)) + 0 * L_envelope + 0.3 * L_paired_chroma
PI05: 0.10 * L_flow + 0.03 * mean(abs(mean(Y_aux)-0.6)) + 0 * L_envelope + 0.3 * L_paired_chroma
```

前向可能仍计算 envelope loss，但乘 0 后不贡献总目标；`chroma_prior_enabled=true` 与 `chroma_prior_weight=0` 必须区分。

## 4. extends、优先级与完整有效配置

**Confirmed：** `starVLA/starVLA/config_loader.py::load_config/_load_config:14–53` 解析相对路径（相对于声明 extends 的 YAML 所在目录），递归合并 bases，再合并 child；多个 bases 从左到右合并；检测循环。不是相对于 shell 的当前目录。

```text
configs/base/rawvla.yaml
├─ configs/base/rawvla_libero.yaml
│  ├─ configs/libero/qwen3_oft.yaml
│  ├─ configs/libero/pi0.yaml
│  ├─ configs/libero/pi05.yaml
│  └─ configs/libero/optional/{qwen3_pi,wm4a_cosmos,wm4a_wan}.yaml
└─ configs/base/rawvla_robotwin.yaml
   └─ configs/robotwin/{pi0,pi05}.yaml
```

各 child/base 第一行或第二行给出上述 extends。最高优先级为 CLI dotlist：`train_starvla.py:985–995` 在 load_config 后 `OmegaConf.merge(cfg,cli_cfg)`，再执行兼容字段迁移；`trainer_tools.py::normalize_dotlist_args:31–54` 支持 `--x.y value`、`--x.y=value` 和裸 `--flag`（true）。**注意没有 `--` 前缀的孤立 `x.y=value` 会被此 parser 丢弃**。证据 `trainer_tools.py:42–53`。`share_tools.py::apply_config_compat:418–449` 列出的迁移是 action model shape/旧字段和 version_id，未把曝光 .01 改成 .001。

**Confirmed（静态合并结果）：** 本次用 PyYAML 递归按上述 merge 顺序读取全部八个 child，仅做 YAML 分析；环境没有 OmegaConf/Torch，因此未运行 production loader 或训练。输出如下，`paired` 指配置设 rgb_cache_root，`prior` 与 `paired` 为两个不同损失。

| Child | action weight | exposure weight | chroma prior weight | paired chroma weight | exposure target | optimizer steps / warmup | raw_frontend LR | weight decay |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| LIBERO Qwen3-OFT | .20 | .01 | .001 | 0（缺省） | .48 | 2000 / 100 | 1e-4 | 1e-8 |
| LIBERO PI0 | .15 | .01 | .001 | 0（缺省） | .48 | 2000 / 100 | 1e-4 | 1e-8 |
| LIBERO PI05 | .15 | .01 | .001 | 0（缺省） | .48 | 2000 / 100 | 1e-4 | 1e-8 |
| LIBERO optional Qwen3-PI | .0015 | .01 | .001 | 0（缺省） | .48 | 2000 / 100 | 1e-4 | 1e-8 |
| LIBERO optional WM4A-Cosmos | .01 | .01 | .001 | 0（缺省） | .48 | 2000 / 100 | 1e-4 | 1e-8 |
| LIBERO optional WM4A-Wan | .02 | .01 | .001 | 0（缺省） | .48 | 2000 / 100 | 1e-4 | 1e-8 |
| RoboTwin PI0 | .02 | .03 | 0 | .3 | .60 | 1000 / 50 | 5e-5 | 1e-10 |
| RoboTwin PI05 | .10 | .03 | 0 | .3 | .60 | 1000 / 50 | 5e-5 | 1e-10 |

表中来源：`configs/base/rawvla.yaml:34–52,57–80`；`configs/base/rawvla_robotwin.yaml:35–60`；`configs/libero/qwen3_oft.yaml:22–25`，`pi0.yaml:27–30`，`pi05.yaml:27–30`；三个 optional YAML 各末尾 `trainer.action_loss_weight`（qwen3_pi:46–49、wm4a_cosmos:45–48、wm4a_wan:28–31）；`configs/robotwin/pi0.yaml:15–17`，`pi05.yaml:14–15`。

共同有效 loss 设置还包括：`two_sided_l1`、STE exposure clamp、chroma prior enabled、output_envelope、mean_log descriptor、zero bias、margin=.1，lower=[.025407744850963355,-.35791672915220263]，upper=[.25704274773597713,-.0317848339676857]；domain target override 默认空、domain exposure multiplier 默认1；paired supervision beta=.05、有效目标均值亮度 [.02,.98]（RoboTwin显式设置，其余仅有目标才使用）。

共同训练设置来自 `configs/base/rawvla.yaml:57–80`：batch/device=1，YAML accumulation=8，AdamW betas=(.9,.95)，eps=1e-8，cosine_with_min_lr 到1e-6，clip=1，frozen_modules_eval=true，只保存可训练项。**但 accumulation 的运行时生效有下述限制。**

### 梯度累积不能只看 YAML

**Confirmed（代码读取范围）：** `train_starvla.py:51–52` 在解析 YAML 前构造 `DeepSpeedPlugin()` 和 `Accelerator(deepspeed_plugin=...)`，未传 `cfg.trainer.gradient_accumulation_steps`；`TrainerUtils.setup_distributed_training:318–328` 只 `accelerator.prepare`。`_calculate_total_batch_size:209–215`、日志 `:629–632` 使用的是 `accelerator.gradient_accumulation_steps`。本训练路径未读取 YAML 的该键来设置 runtime accumulation。

**Unknown：** 作者实际训练的 runtime accumulation。可能由 Accelerate/DeepSpeed 外部配置或环境变量设为8；本次无外部配置/运行日志，不能把 YAML 中的8当作已验证有效值。正式复现应显式设置 `ACCELERATE_GRADIENT_ACCUMULATION_STEPS=8`，同时确认 DeepSpeed 配置及启动日志一致；smoke test 用1。将总损失统一除以 accumulation 会同时缩放 action/brightness/chroma，不能抵消 brightness 单项十倍相对权重。

## 5. 0.001 与 0.01：数值证据及不能确定的部分

给同一批图像 `mean(Y)=.20`、target=.48：当前 two_sided_l1 原始标量=.28；`.01*.28=.0028`，论文 `.001*.28=.00028`，相差十倍。Chroma 原始项及 Qwen3-OFT action 的系数没有同步放大十倍，因此总目标也不是一个等比例缩放。

**Confirmed：** 对 README 的 Qwen3-OFT 启动例子，`README.md:188–204` 只 CLI 覆盖 dataset cache_root，未覆盖曝光或 chroma weight，实际选中 `.20*L1_action+.01*L_bright+.001*L_chroma`。该入口 RGB cache_root 缺省为空，故没有 paired loss。

其它可确认的论文差异：

- LIBERO PI05：Table6 `λ_a=.10`，当前 `configs/libero/pi05.yaml:30` 为 `.15`。
- 论文 D.1 为 PI0/PI05 weight_decay=1e-10；当前两个 LIBERO PI 配置未覆盖共享1e-8（RoboTwin才设1e-10）。
- RoboTwin 如第1、3、4节所述，aux、steps、LR、warmup 均与 D.1/D.3 不同；当前 PI0 action=.02，也与 Table6 PI0=.15不同。
- `baselines/rawvla/optimal_backbone_configs.yaml:1–5,18–24,33–34` 也记载曝光=.01、PI05=.15、selected_step=2000；这是发布文件中对当前数值的独立记录，不能证明实际训练日志。`rg -n optimal_backbone_configs -g '*.py' -g '*.sh'` 无命中，故此文件是说明/选择元信息，不会由 README training 入口自动继承。
- 子模块旧示例 `starVLA/examples/simBenchmarks/LIBERO/train_files/starvla_rawvla_bench_light_rawvla.yaml:10,33,59,80–105` 是 QwenGR00T+lerobot 的另一套100,000-step配置，不能用它代替 README指定的 NPZ 主配方。

**Unknown：** 哪份发布配置对应论文表格中的每个最终checkpoint；是否存在作者未发布的 CLI override；实际运行中 exposure/chroma/action 的数值、梯度范数、runtime accumulation；应以作者 run config.full.yaml、命令记录、checkpoint 和论文勘误核对。此阶段不能把任何一个差异命名为已证实的训练 Bug。

## 6. 下一阶段可执行的只读核对与最小运行建议

下面是未来命令，未在本阶段运行。先在已安装环境解析完整配置，不启动训练、不加载权重：

```bash
export RAWVLA_ROOT=/path/to/RawVLA
cd "$RAWVLA_ROOT/starVLA"
PYTHONPATH="$RAWVLA_ROOT/starVLA:$RAWVLA_ROOT" python - <<'PY'
from starVLA.config_loader import load_config
from omegaconf import OmegaConf
c=load_config('../configs/libero/qwen3_oft.yaml')
print(OmegaConf.to_yaml(c,resolve=False))
# 用resolve=False避免尚未准备权重路径环境变量时失败。
# 下一阶段再按trainer:985–990的顺序合入CLI dotlist并保存真实完整结果。
PY
```

CPU RAW 前端 shape/梯度测试可用现有 `python -m baselines.rawvla.smoke_test`（证据 `baselines/rawvla/smoke_test.py:1–76`）。注意无 Torch 时它会输出 skipped 并正常返回，**不能只用退出码0验收**；先检查 `python -c 'import torch; print(torch.__version__)'`，并确认测试确实执行。此测试不含 frozen VLA action 链，只验证 ISP 自身。

完整 VLA 一个 batch 需要本地 backbone、tokenizer、stats、一个合法 NPZ、可容纳 frozen VLA 反传的 GPU。可在下一阶段把现有 Qwen3-OFT 命令限制到一个 optimizer step，并设 logging=1取得 component gradients；不修改官方文件：

```bash
export RAWVLA_ROOT=/path/to/RawVLA
cd "$RAWVLA_ROOT/starVLA"
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1 \
PYTHONPATH="$RAWVLA_ROOT/starVLA:$RAWVLA_ROOT" \
accelerate launch --num_processes 1 starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/qwen3_oft.yaml \
  --datasets.vla_data.cache_root /path/to/single_npz_parent \
  --trainer.max_train_steps 1 --trainer.logging_frequency 1 \
  --trainer.num_warmup_steps 0 --trainer.save_interval 1000000 \
  --run_id audit_one_step --run_root_dir /path/to/audit_outputs
```

此命令会产生一个真实训练step和最终checkpoint，因此仅供下一阶段，不属于本次执行；不能拿该step评估论文精度。不要把它加上 `.001` override 后称“完全原样复现”；若比较论文配方与发布配方，应做两个有独立命名、独立配置快照的实验。

验收：读取输出 `config.full.yaml`（`train_starvla.py::_save_initial_configs:245–264`）、启动日志的 runtime accumulation；`total_loss` 与各 weighted loss求和一致，值有限；`grad_norm/action_to_isp` 与 `grad_norm/weighted_action_to_isp` 非零；曝光/颜色分支有对应非零grad，所有 VLA 参数 frozen。指标产生位置 `train_starvla.py:669–708,872–911`。若缺梯度，先区分目标当前处于envelope内导致零loss、clipping饱和、图像转换detach、freeze误匹配、前端未启用；若权重读错，核对入口YAML、CLI前缀、extends和run目录快照，不凭文件名判断。

## 7. 下一阶段“发布默认”与“按论文文字”的隔离方法

**Confirmed（配置/公式核对）；所有命令均为建议，未执行。** 发布默认模式使用第4节表中的原始配方；按论文文字模式用 CLI 明确覆盖，并在 run_id 写 `paper_text`。以下覆盖能使已定位的显式超参数接近论文文字，但不能替代作者最终训练日志，也不代表已复现论文结果。

| 主配置 | 保留发布默认所需操作 | 按论文文字额外 CLI 覆盖 |
| --- | --- | --- |
| LIBERO Qwen3-OFT | 不覆盖 loss | `--framework.raw_frontend.exposure_loss_weight 0.001` |
| LIBERO PI0 | 不覆盖 loss/decay | 上项，加 `--trainer.optimizer.weight_decay 1e-10` |
| LIBERO PI05 | 不覆盖 loss/decay | 曝光 .001，加 `--trainer.action_loss_weight 0.10 --trainer.optimizer.weight_decay 1e-10` |
| RoboTwin PI0 | 不覆盖 loss/paired目标 | 见下列共享覆盖，加 `--trainer.action_loss_weight 0.15` |
| RoboTwin PI05 | 不覆盖 loss/paired目标 | 见下列共享覆盖；action 默认 .10 已同 Table6 |

RoboTwin 论文文字模式的共享覆盖（命令片段）：

```bash
--framework.raw_frontend.exposure_target 0.48 \
--framework.raw_frontend.exposure_loss_weight 0.001 \
--framework.raw_frontend.chroma_prior_weight 0.001 \
--framework.raw_frontend.chroma_supervision_weight 0.0 \
--datasets.vla_data.rgb_cache_root '' \
--trainer.max_train_steps 2000 \
--trainer.num_warmup_steps 100 \
--trainer.learning_rate.base 0.0001 \
--trainer.learning_rate.raw_frontend 0.0001
```

清空 `rgb_cache_root` 用于保证不载入训练配对目标；checkpoint selection 若要利用 paired RGB，应另做无梯度选择流程。当前论文/代码未提供最终run的一一映射，不能保证 envelope 边界、PI policy/task微调版本、数据量化版本等也与作者历史实验相同。RoboTwin 的 max_exposure_ev=10、gray_preserving_ccm=true 也属于当前 benchmark差异；论文未提供足够逐benchmark对照来确认是否应统一为LIBERO默认，故此处不擅自覆盖。

## 8. 训练入口的 DeepSpeed 与 CUDA 前提

**Confirmed：不能把 README 命令理解成只装 `environment-libero.yml` 即一定可启动。**

- `train_starvla.py:51–52` 无条件传入 `DeepSpeedPlugin`；`environment-libero.yml` 虽 pin Accelerate=1.5.2（第20行），文件无 deepspeed 依赖项；`starVLA/requirements.txt:23` 才单独列出 `deepspeed==0.16.9`。因此训练环境须额外核对 `import deepspeed` 和 CUDA兼容性。
- 官方 [Accelerate v1.5.2 accelerator.py:336–355](https://github.com/huggingface/accelerate/blob/v1.5.2/src/accelerate/accelerator.py#L336) 明确：传入 plugin 会设置 `ACCELERATE_USE_DEEPSPEED=true`；缺 DeepSpeed 抛 ImportError。这是 pin版本源码直接证据，不是推测。即使使用一个进程，也不能通过 `ACCELERATE_USE_DEEPSPEED=false` 绕过代码传入的plugin。
- 官方 [Accelerate v1.5.2 dataclasses.py:1124–1154](https://github.com/huggingface/accelerate/blob/v1.5.2/src/accelerate/utils/dataclasses.py#L1124) 的 `DeepSpeedPlugin.__post_init__` 读取 `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`、默认 zero_stage=2，并读取 `ACCELERATE_DEEPSPEED_CONFIG_FILE`。同版 [accelerator.py:513–519](https://github.com/huggingface/accelerate/blob/v1.5.2/src/accelerate/accelerator.py#L513) 也读取 accumulation环境变量。因此显式env是有源码支持的设置办法；仍应排除外部DS JSON覆盖或不匹配并查看实际日志。
- 当前训练 `_train_step` 强制 `torch.autocast("cuda",dtype=bfloat16)`（第639行）；本文件本地定义 `setup_optimizer_and_scheduler:89–99` 使用 AdamW `fused=True`，遮蔽了import的同名helper（helper本身会检查cuda可用）。因此 CPU 路线应停留在数据检查/ISP前端测试；完整训练入口不应承诺CPU可跑。

GPU训练前的只读预检（未来执行）：

```bash
python - <<'PY'
import torch, accelerate, deepspeed, transformers, omegaconf
print('versions',torch.__version__,accelerate.__version__,deepspeed.__version__,transformers.__version__)
print('cuda',torch.cuda.is_available(),'bf16',torch.cuda.is_bf16_supported())
PY
```

若缺DeepSpeed，应先用单独环境补齐和验证构建，不在正式训练时临时修；若单GPU分布式初始化报错，需核对launcher是否设置正确rank/world/master变量及DS后端。未在本阶段验证single-process plugin/distributed初始化，故最小训练命令可执行性为 **Partially Confirmed**，不得报告成已通过。
