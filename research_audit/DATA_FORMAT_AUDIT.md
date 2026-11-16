# DATA_FORMAT_AUDIT — Audit A

审计日期：2026-11-16（Asia/Shanghai）；只读静态审计。根仓库 `ba9c56b04da2c4931b70d8844710f5928d43775b`；starVLA 实际检出 `57d508cb180d0510ac9060358009c0f6ffaac420`。本文路径相对 `/workspace/work/RawVLA/`。`starVLA/` 内的行号属于上述子模块 SHA，不能用根仓库的 blob 链接替代。

证据等级：**Confirmed** = 已直接核对当前源码或小型元数据；**Partially Confirmed** = 已建立静态链路，但尚未用发布样本/完整依赖运行核验；**Unknown** = 公开资料或当前只读范围无法确定。没有下载任何 NPZ、数据集或权重，没有运行模型或训练。

## A0. 核心结论

| 结论 | 等级 | 证据 |
|---|---|---|
| 模型输入是三通道线性 RGB 顺序的 pseudo-RAW，不是单通道 Bayer。 | Confirmed | `configs/base/rawvla.yaml:18-19` 的 `source_key/input_format`；`starVLA/starVLA/model/modules/raw_frontend/registry.py:315-325` 的 `RAWVLAFrontend.forward`；论文 Appendix A [链接](https://arxiv.org/html/2609.37530v1#A1)。 |
| LIBERO 先生成 10-bit 中间量，发布/回放缓存再降为 8-bit；训练读出后除以 255。因此“pseudo-RAW10 生成”和“模型保留完整 10-bit 输入”不是同一结论。 | Confirmed（源码）；Partially Confirmed（发布二进制） | `experiments/simulation/libero/rawvla_bench/raw.py:105-106,151-163`；`scripts/replay/replay_libero_train_rawvla_light_npz.py:344-345,536-545`；Dataset `rawvla_npz_datasets.py:263-266`；HF 卡见 A1。 |
| RoboTwin 的 HDR 主路径直接保留 float32 [0,1]，不会经过 RAW10 或 uint8；同一 camera 函数也有 legacy RAW10 路径，不能仅凭变量名 `raw_linear` 判定来源。 | Confirmed | `third_party/RoboTwin/envs/camera/camera.py:352-376`；`hdr_linear_to_raw_float`，`raw_rgb10.py:206-230`；回放脚本 `replay_robotwin2_paired_lighting_training.py:173-181,225-230`。 |
| `raw_burst_float32` 是 Dataset 动态构造的样本键，不是 NPZ 中预存字段。每个 view 的形状为 `[6,6,H,W,3]`，两个 6 分别是 recurrent endpoints 与相邻帧 burst。 | Confirmed | `RAWVLALightNPZDataset.__getitem__`，`starVLA/starVLA/dataloader/rawvla_npz_datasets.py:224-230,261-286`。 |
| 默认 LIBERO 不消费配对 RGB；手动加 `rgb_cache_root` 也不能直接得到现有发布字段的色度目标，因为 LIBERO 分支把 `rgb_keys` 设成 `raw_keys`。RoboTwin 分支的字段映射正确。 | Confirmed（静态映射）；Partially Confirmed（运行影响） | Dataset `:244-269,290-293`；LIBERO RGB writer `replay_libero_train_rawvla_light_npz.py:557-565`；`configs/base/rawvla_libero.yaml:4-10`。不可据此声称默认主训练已经失败。 |
| 公开数据说明与当前 producer/consumer 的基础 dtype 和 RAW 字段一致；论文描述还不足以证明发布 LIBERO 缓存保留 10-bit，且 RGB 配对读取存在上述例外。 | Partially Confirmed | A1–A7 的逐段证据；没有读取发布 NPZ 的数组头或内容。 |

## A1. Hugging Face 发布物与审计边界

官方数据仓库：[ToferFish/RawVLA-Bench](https://huggingface.co/datasets/ToferFish/RawVLA-Bench)。本次 GET `https://huggingface.co/api/datasets/ToferFish/RawVLA-Bench` 只读取 JSON 元数据，命令输出为：

```text
sha = 8586c594fff9ce38580e29745f074e82e96e74f7
lastModified = 2026-09-29T11:28:22.000Z
gated = False; private = False
usedStorage = 713689384758
siblings = 4810; NPZ files = 4806
data/libero/raw: 1771; data/libero/rgb: 1771
data/robotwin2/raw: 632; data/robotwin2/rgb: 632
```

**Confirmed**：上述文件数由 API 文件名计数得到；`usedStorage` 是服务返回的仓库存储统计，不等于本次下载量或单一快照实际下载大小。仅取小型 [README](https://huggingface.co/datasets/ToferFish/RawVLA-Bench/blob/8586c594fff9ce38580e29745f074e82e96e74f7/README.md) 和 API JSON，没有获取任何 NPZ。HF README 行号按本次取回的文件：53–60 声明两子集 dtype/配对数，88–95 与 114–123 声明字段，139–140 提醒 episode NPZ 可能较大。

**Partially Confirmed**：HF 卡对数组的声明与源码相符，但不能把卡片当作实际二进制数组检查；所有发布文件是否逐个 dtype/shape/配对对齐，仍需后续抽样乃至全量独立验证。HF 卡 82–86 还报告 LIBERO 模拟器版本导致成功复验差异，这也是复现环境必须锁定的原因；本审计没有重新运行成功率复验。

## A2. LIBERO 的完整生成链

1. **Confirmed**：`_replay_episode` 读取 HDF5 中的 `actions/states/dones/robot_states`，创建 clean 与 target-light 两个环境。去除 no-op，并以 clean 场景的物理状态驱动 RAW 渲染分支。相机输出先作 `[::-1, ::-1]`，即空间旋转 180°，没有 RGB/BGR 交换。证据：`scripts/replay/replay_libero_train_rawvla_light_npz.py:233-251,273-325`。
2. **Confirmed**：每个 agent/wrist target-light RGB8 送入 `make_rawvla_observation`。`_rgb8_to_exposed_raw10` 将 RGB/255 → inverse smoothstep → inverse sRGB → 除通道增益 `[1.8,1.0,1.7]` → 曝光增益 → clip → round(x*1023) → uint16。证据：`starVLA/examples/simBenchmarks/LIBERO/eval_files/raw_rgb10_unprocessing.py:31-36,50-86`；根模块 `raw.py:25-30` 优先 import 该实现，并在 `:72-81` 提供等价 fallback。
3. **Confirmed**：`make_rawvla_observation` 依次使用 `apply_full_well_saturation`、`add_sensor_noise`；噪声方差为 `4e-4*x + 1e-5`，通过 trajectory/view/frame 相关 seed 确定性采样，再量化回 10-bit。`representation='raw'` 最终调用 `raw10_to_view8`，执行 `round(raw10*255/1023).astype(uint8)`；`default_isp` 则重处理同一 noisy RAW10。证据：`experiments/simulation/libero/rawvla_bench/raw.py:90-96,105-131,134-164`。
4. **Confirmed**：`_process_refs` 实际写 RAW 文件时只保留 `agentview_raw_uint8/wrist_raw_uint8` 和动作等公共字段，RGB 分支另写 `agentview_rgb_uint8/wrist_rgb_uint8`。内存返回字典还包含 `*_target_rgb_uint8` 等，但不等于落盘 NPZ 字段，不能照抄 `_replay_episode` 返回值当 schema。证据：`replay_libero_train_rawvla_light_npz.py:357-370` 对比 `:536-567`。
5. **Confirmed**：unprocessing 和 frame noise 在 256×256 renderer 分辨率执行，policy resize 在后面；评估侧注释也明确这一顺序。证据：回放脚本 `:523-529` 的 `source_shape/render_order`；`starVLA/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py:430-459`。论文 Appendix A 的概括式“resized to VLA input resolution”和 B.1 的 256×256 描述应结合此实现阅读；若要逐像素重现，应遵从实际代码顺序，不能自行提前 resize。

**论文对应关系（Partially Confirmed）**：论文 [Appendix B.1](https://arxiv.org/html/2609.37530v1#A2.SS1) 明确三通道 RAW10、饱和及噪声生成，这与中间处理相符；当前公开 producer 还有 RAW10→uint8 落盘转换。没有论文实验专用 commit 或运行产物证明论文训练是否正是该 uint8 缓存。因此“链路可解释”已确认，“与论文逐比特一致”仍未知，不能仅凭名称宣称 Bug。

额外 CPU 算术检查（只用 Python 标准库枚举，不运行官方模型）：1024 个 RAW10 整数经上述映射只得到 **256 个不同值**；`u8/255` 对 `u10/1023` 的最大绝对误差约 **0.0019550342**。这证明除以 255 是正确的 uint8 还原范围，但不能恢复丢失的 10-bit 精度。

## A3. RoboTwin float32 的来源

| 环节 | 实际行为 | 等级与证据 |
|---|---|---|
| HDR 相机 | 当 `hdr_raw_white_level>0`，读取 `HdrColor[:,:,:3]`，移除固定 WB，再除 white level 和 clip 到 [0,1]；不逆 gamma、不逆 tone。 | Confirmed：`camera.py:352-355`；`raw_rgb10.py:206-230` 的 `hdr_linear_to_raw_float`。 |
| 饱和/噪声 | 浮点饱和压力在 RAW 域，噪声后仍 float；RAW10 与 view8 只是兼容输出。 | Confirmed：`camera.py:355-367`；`raw_rgb10.py:233-240,297-320`；噪声数值由 replay entry 提供，`replay_robotwin2_paired_lighting_training.py:173-181`。 |
| legacy 路径 | white level 不正时，从 Color buffer unprocess 为 RAW10，再以 `/1023` 产生 float；因此 float dtype 本身不能证明 HDR 未量化。 | Confirmed：`camera.py:369-376`。 |
| 配对 replay | clean 环境执行动作；每次取图前复制 actor/articulation 物理状态到 target-light 环境；从后者 `raw_linear` 取 noisy float，同时关噪声重读得到 clean RAW。 | Confirmed：`replay_robotwin2_paired_lighting_training.py:201-235`，`copy_physics_state :34-80`。 |
| 写 NPZ | `build_arrays` 将各 view 的 noisy/clean RAW 分别写成 `*_raw_float32`/`*_clean_raw_float32`；RGB 另写成 uint8。 | Confirmed：同文件 `:83-128`。HF 卡仅列 noisy RAW 必需字段，不承诺 clean RAW 必定在发布文件中。 |

**Confirmed**：训练 manifest builder 的 `RAW_WHITE_LEVEL=3.5`、`RAW_SENSOR_SHOT_NOISE=2.5e-5`、`RAW_SENSOR_READ_NOISE=3.90625e-8`，与通用 camera helper 的默认噪声不同。builder 会把这三个常量写入每个 entry，paired replay 再传给 camera。证据：`scripts/replay/build_robotwin2_lighting_train_manifest.py:19-22,127-130,149-156`；replay `:173-181`。因此本路径的设计噪声方差为 `2.5e-5*x + 3.90625e-8`。

论文 [Appendix B.2](https://arxiv.org/html/2609.37530v1#A2.SS2) 的 HDR/float32 存储描述与此主路径一致（**Partially Confirmed**：未读取发布数组及其白电平 metadata）。噪声方差与白电平仍应从选定 NPZ 的 `metadata_json` 复核，不应只用某个函数的默认值判断实际发布设置。

## A4. NPZ schema：哪些字段真实被写入/读取

以下是 producer 写入字段，已由 consumer 交叉核对；发布二进制未读取，所以“所有发布文件确实含这些字段”的证据等级为 **Partially Confirmed**。

| 子集/文件 | 影像字段 | 公共字段与 shape 约束 | 源码证据 |
|---|---|---|---|
| LIBERO RAW | `agentview_raw_uint8`, `wrist_raw_uint8`，`[N,256,256,3]`，uint8 | `actions`, `states`, `robot_states`, `rewards`, `dones`, `metadata_json`；N 为该 episode 保留帧数 | writer `replay_libero_train_rawvla_light_npz.py:536-545`；reader `rawvla_npz_datasets.py:221,235-239,255-266`。 |
| LIBERO RGB | `agentview_rgb_uint8`, `wrist_rgb_uint8`，同帧数与相对路径 | 同上 | writer `:547-567`；HF README `:88-95`。 |
| RoboTwin RAW | `head_camera_raw_float32`, `left_camera_raw_float32`, `right_camera_raw_float32`；producer 还写各 `*_clean_raw_float32`；HWC float32 | `actions`, `robot_states`, `left_endpose`, `right_endpose`, `rewards`, `dones`, `metadata_json` | writer `replay_robotwin2_paired_lighting_training.py:95-128`；reader `rawvla_npz_datasets.py:244-249`。 |
| RoboTwin RGB | `head_camera_rgb_uint8`, `left_camera_rgb_uint8`, `right_camera_rgb_uint8` | 同上；相对 raw root 的路径映射到 rgb root | writer `:110-128`；reader `:137-149,250-253,267-269`。 |

**Confirmed**：`metadata_json` 是一个 NumPy 字符串标量，由 `np.asarray(json.dumps(...))` 写入，用 `json.loads(str(...item()))` 读取；不是 Python pickle dict。`np.load(...,allow_pickle=False)` 是官方 reader。`raw_burst_float32`、`image`、`action`、`lang`、`state`、`rawvla_light`、`rawvla_chroma_target_float32` 均是 Dataset 返回键，不能要求在原始 NPZ 中找这些名字。证据：Dataset `:130-149,238-239,278-294`；上表 writer。

**Confirmed**：`robot_states` 是存储字段但不必是当前模型的状态源。RoboTwin 默认 `state_source_key: actions`、`action_offset: 1`，即 observation t 的 state 取 joint-action vector t，监督 action chunk 从 t+1 开始；LIBERO 默认状态源为 `robot_states`，offset=0。证据：`configs/base/rawvla_robotwin.yaml:14-20`；Dataset `:67-68,231-238`。动作维度、状态语义不能仅按 NPZ key 名推断。

## A5. Dataset 与 DataLoader

**Confirmed**：`RAWVLALightNPZDataset` 递归枚举 `cache_root/**/*.npz`，排序后用 episode-local cache 一次把该文件所有数组载入内存；不是逐帧磁盘 streaming。`rgb_cache_root` 若设置，则要求相同相对路径存在配对文件。证据：`rawvla_npz_datasets.py:39-41,130-149`。`np.load` 对压缩 NPZ 也不能提供普通 NPY 那样的真正 mmap。`_load_episode` 还会加载 writer 可能留下但训练不消费的 clean RAW 等数组；因此内存预算需按整个 episode，而不只是一个 batch 估算。

**Confirmed**：样本索引 i 的随机数为 `seed + i*104729`；没有 lighting weights 时 episode slot 为 `(i//samples_per_episode)%len(paths)`，默认一个 episode 连续处理32个索引；`current` 从该 episode 采样。`virtual_length=100000` 不是 NPZ 文件数。`set_epoch` 不修改采样状态。若配置 lighting weights，会先用 metadata 为路径建立概率。证据：Dataset `:42-71,105-109,208-222`；shared YAML `:5-13`。

**Confirmed**：`collate_fn(batch): return batch`；DataLoader 不默认 stack 全部字典，不统一转为 tensor；返回 `list[dict]`。`build_dataloader` 使用配置的 per-device batch size/workers/pin/prefetch；默认 batch=1、worker=1、prefetch=1、persistent worker=true。证据：`rawvla_npz_datasets.py:14-15`；`starVLA/starVLA/dataloader/__init__.py:36-69`；`configs/base/rawvla.yaml:9-13`。需要纯只读加载时直接调用 Dataset；`build_dataloader` 的 `:66-68` 会在 output_dir 写 `dataset_statistics.json`，不适合无写入的格式检查。

**Confirmed**：图像范围只作 uint8/255，float 原样转 float32，没有 per-image minmax、均值方差或 gamma。动作归一化与图像归一化完全独立：LIBERO Qwen `starvla_minmax` 的前6动作维度映到[-1,1]，gripper `(x+1)/2`；PI0 zscore/delta、PI05 quantile 从各自 stats 文件读取；RoboTwin zscore 与 action offset 以配置为准。证据：Dataset `:151-206,263-269`；`configs/libero/pi0.yaml:7-10`、`pi05.yaml:7-10`。即使只检查一个真实 NPZ 的 Dataset 返回，也要提供相符 stats JSON；文件不存在时 `:90-103` 直接报错，不应伪造 stats 来做真实复现。

## A6. 六帧 burst 与六个 recurrent endpoints

令 current=t，stride=s，Nrec=K=6：

```text
endpoint[j] = clip(t - s*(5-j), 0, N-1), j=0..5
burst[j,k]  = clip(endpoint[j] - (5-k), 0, N-1), k=0..5
```

**Confirmed**：这是 Dataset `:224-230` 的直接展开；不能把六个 endpoint 当六个相邻帧，也不能把每个 burst 的帧间隔误设为 s。靠 episode 起点时重复首帧，不跨 episode。终点 action chunk 超界时重复末动作，证据 `:231-234`。

| current=50 | endpoint 列表 | terminal burst |
|---|---|---|
| Qwen LIBERO s=8 | 10,18,26,34,42,50 | 45,46,47,48,49,50 |
| PI0/PI05 s=5 | 25,30,35,40,45,50 | 45,46,47,48,49,50 |

**Confirmed**：LIBERO base stride=8；PI0 和 PI05 leaf YAML 覆盖为5；RoboTwin PI base 也是5；上述数值由公式用 Python 标准库重算。证据：`configs/base/rawvla_libero.yaml:8`、`configs/libero/pi0.yaml:7`、`pi05.yaml:7`、`configs/base/rawvla_robotwin.yaml:12`。

**Confirmed**：frontend 将每 view 转成 channel-first，然后将 batch 与 views flatten 成 `B*V`：`[B*V,6,6,3,H,W]`。registry 中依次调用6次 model，每次 `state/theta` 传给下一次，不在此处 detach；`RAWVLA.forward` 只用当前 burst 最后一帧估计 operating point，用整个6帧 burst 做 FFT merge，最后 structured color/tone 输出 RGB。训练只把第6次返回的 RGB 送给 VLA。证据：`qwen_tensor_vision.py:28-68,164-170`；`registry.py:317-338,522`；`baselines/rawvla/rawvla.py:1099-1128`。与论文 Appendix D.3 的双时间轴和末端输出描述相符。

**Confirmed（LIBERO公开客户端）**：在线评估每个 camera step 更新 burst deque，不足6帧时重复首帧；policy replanning 时 `_with_raw_frontend_history` 再收集 recurrent burst deque，不足6个 endpoint 时重复最早 burst，`reset` 清空历史。这避免把 camera cadence 和 policy cadence 混为一谈。证据：`starVLA/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py:430-459,495-498`；`model2libero_interface.py:91-124`。本条不替代 Audit C 对其他 runner 的协议检查。

## A7. RAW Frontend → RGB → VLA

```text
NPZ [N,H,W,3]
  → RAWVLALightNPZDataset: 每 view [Nrec,K,H,W,3] float32 [0,1]
  → DataLoader collate: list[dict]，不自动 tensor stack
  → select_raw_frontend_images(source_key=raw_burst_float32)
  → _as_raw_frontend_tensor: 每 view [Nrec,K,3,H,W]
  → stack views/batch: [B*V,Nrec,K,3,H,W]
  → RAWVLAFrontend.forward: recurrent ×6；每次 burst ×6
  → RAWVLA.forward: FFT merge → exposure/WB/CCM → tone
  → 末端 RGB [B*V,3,H,W] float32 [0,1]
  → policy adapter：resize/pack 与该 backbone 的标准归一化
  → frozen VLA/action head
```

| 接口 | 已核实行为 | 等级/证据 |
|---|---|---|
| RAW source 选择 | 优先使用配置 `source_key`；若缺失，依次尝试 fallback，最后可能读 `image`。最小复现应断言 `raw_burst_float32` 存在，防止无声退化为单图。 | Confirmed：`qwen_tensor_vision.py:113-144`。 |
| 通道和范围 | HWC→CHW，只做 permute，没有 BGR；float 必须已在[0,1]，uint8 才除255；其他整数拒绝，NaN/Inf/越界拒绝。 | Confirmed：同文件 `:28-68`；`image_contract.py:38-70`。 |
| RGB 输出合同 | Tensor、RGB channel-first、float32、finite、[0,1]；断言只 detach 检查副本，返回原 tensor，保留计算图。 | Confirmed：`image_contract.py:18-35`；`qwen_tensor_vision.py:190-203,292-297`。 |
| Qwen3-OFT | PIL `image` 用于 processor 的空间/token layout，之后用原始 RAW 经可微 frontend 得到的 RGB 替换 `pixel_values`；bicubic resize→clip→`(x-image_mean)/image_std`→patch packing。默认 mean `[0.48145466,0.4578275,0.40821073]`，std `[0.26862954,0.26130258,0.27577711]`，但优先读 image_processor 实际属性。 | Confirmed：`QwenOFT.py:172-205`；`qwen_tensor_vision.py:71-88,210-254,257-342`。没有用 PIL 量化后的图作为可训练 RGB 像素输入。 |
| PI0/PI05 | `_apply_raw_frontend` 把 `image` 替换为可微 RGB tensor；`_image_to_tensor` bilinear 等比例resize/pad，并明确执行 `2*x-1`；之后 embed_image。PI05 继承同一图像路径。 | Confirmed：`VLM4A/PI0.py:254-295,297-324,356-386,503`；`VLM4A/PI05.py:50-58`。不要根据 `:273-274` 注释误以为 adapter 未做[-1,1]映射，以实际 `:276` 返回值为准。 |

## A8. 已确认的一致性与仍未确认的部分

| 对照点 | 判断 | 证据等级与原因 |
|---|---|---|
| 三通道非Bayer、RGB顺序、固定输入范围 | 论文定义与代码接口一致 | Confirmed（接口）；实际每个发布样本仍未验。 |
| LIBERO uint8 的读写 | HF 卡、producer、Dataset 一致，均按 uint8/255 传输 | Confirmed（声明和代码）。 |
| LIBERO RAW10 保真 | 仅中间生成是10-bit，落盘再降8-bit；论文未在相应段明确这一转换 | Partially Confirmed；无法确认论文实验实际数据 commit/缓存。 |
| RoboTwin HDR float32 | 论文、HF声明、主路径代码一致 | Partially Confirmed；未抽取发布 metadata/数组检查 HDR white level 与量化格点。 |
| 双时间轴与边界padding | 论文和当前 Dataset/registry 一致 | Confirmed（静态）；没有运行真实episode验证索引。 |
| LIBERO配对RGB可直接作为训练目标 | 当前 leaf/base 默认不用；reader字段不匹配，手动开启也不会生成目标键 | Confirmed（静态）；应作为后续修复/回归检查议题，不修改本次官方代码。 |
| RoboTwin clean RAW附加字段 | 当前producer会写，consumer会加载但不直接使用 | Confirmed（源码）；发布文件是否保留仍 Unknown，HF卡未穷举。 |
| 论文结果逐数值复现 | 尚无依据 | Unknown；缺真实样本核验、冻结权重、训练运行日志及论文专用版本映射。 |

## A9. 最小数据检查建议（下一阶段，未执行）

优先从一个已获准准备的 LIBERO NPZ 做 **CPU 数组与索引核验**，不加载 VLA：使用 `np.load(path,allow_pickle=False)` 列出 `files`、每个数组 `dtype/shape/min/max`、解码 `metadata_json`，检查图像N与actionsN一致、两个view同N、RAW dtype确为uint8，记录SHA256；再按相同相对路径核对RGB配对 actions/states。资源至少按解压后整个episode规划，256×256 双view每帧约0.375MiB图像，N=200仅RAW图像约75MiB，加RGB与临时副本后建议空闲RAM≥1–2GiB。此数值是 shape 算术估算，不是实测最小内存。

再给 `RAWVLALightNPZDataset` 准备该 backbone 的真实 stats 文件，固定 seed/index，打印并独立重算current/endpoint/burst/action indices，断言每view为`[6,6,H,W,3]`与float32[0,1]；只调用Dataset和轻量DataLoader，不走会写dataset_statistics的统一builder。验收时同时断言 source key，以免触发fallback。若只有假造stats的合成fixture，应标为接口测试，不能当作实际动作归一化验证。

RoboTwin重复同一流程，额外确认float32未被误除255、3个camera顺序、`state_source_key=actions`和action_offset=1；对比浮点值是否仅落在1/1023量化格点，以排除误走legacy路径。数据路径错误、stats key错误、配对相对路径缺失、字段名映射差异、越界float均应在加载阶段解决，随后才做CPU单帧frontend。完整分阶段命令与验收标准见主Agent的 `MINIMAL_REPRO_PLAN.md`。
