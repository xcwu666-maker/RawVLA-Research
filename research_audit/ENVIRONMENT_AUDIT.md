# RawVLA Phase 1 — 环境、子模块与预训练权重审计

审计日期：2026-11-16（Asia/Shanghai）。独立 Agent C；根 SHA `ba9c56b04da2c4931b70d8844710f5928d43775b`。路径相对根仓库；模型大小来自 Hugging Face 只读 API 的 `siblings[].size`，不是实际下载。**未安装环境、未下载权重/数据/仿真assets、未运行训练或仿真**。Confirmed=源码/元数据直接证明；Partially Confirmed=链路存在但运行/来源链不完整；Unknown=公开材料或本地证据不足。

## D1. 两套环境的职责与依赖

| 工作 | 应用环境 | 已确认核心依赖 | 源码证据 |
|---|---|---|---|
| LIBERO RAW-VLA训练 | `starvla-libero` | Python3.10、Torch2.6.0+cu124、torchvision0.21.0、Transformers4.57.0、Accelerate1.5.2、NumPy1.26.4、OmegaConf、safetensors、sentencepiece | `environment-libero.yml:5-40` |
| LIBERO仿真评估 | 同上 | MuJoCo3.2.3、robosuite1.4.0、bddl1.0.1、gym0.25.2、LIBERO源码、EGL | yml:41-44；`ENVIRONMENTS.md:18-50` |
| RoboTwin NPZ RAW-VLA训练 | **仍是 `starvla-libero`** | 三视角NPZ读取、PI0/PI0.5 PyTorch policy；离线训练无需创建SAPIEN场景 | `README.md:56-63`；`configs/base/rawvla_robotwin.yaml:4-34` |
| RoboTwin policy server | `starvla-libero` | StarVLA/backbone、websocket；和simulator可用不同解释器 | `README.md:70-71,100-103` |
| RoboTwin仿真客户端 | `robotwin` | Python3.10、Torch2.6.0+cu124、SAPIEN3.0.0b1、MPLib0.2.1、SciPy1.10.1、skimage0.21.0、gymnasium0.29.1、trimesh4.4.3、Open3D0.18.0、CuRobo0.7.8、warp-lang1.12.0 | `environment-robotwin2.yml:5-47`；`third_party/RoboTwin/script/_install.sh:63-84` |

表中结论均 **Confirmed（依赖声明与职责）**；本阶段不声称在当前机器上运行验证过这些组合。作者文档将它称为已验证的环境，不能据此将本机状态也标记为验证通过。

**Confirmed** — `ENVIRONMENTS.md:3-5` 要求环境分离。环境文件不是完全锁定：大量包未指定版本，pip/Conda resolver及平台差异仍会影响可重复性。`environment-libero.yml` 声明的是CUDA wheel，不等于安装了可编译CuRobo的系统CUDA toolkit或宿主机驱动。`ENVIRONMENTS.md:14-16` 明示驱动、Vulkan/EGL、编译器和assets依赖宿主机。

**Confirmed（训练入口依赖缺口）** — `starVLA/starVLA/training/train_starvla.py:35,51-52` 在模块顶层构造 `DeepSpeedPlugin()` 并传给 `Accelerator`；`environment-libero.yml` 没有声明 `deepspeed`，同样未声明入口顶层import的 `wandb`（train_starvla.py:34）。仅按此YML创建环境不等于训练入口依赖闭合。后续Environment Agent须在独立环境核实兼容版本与安装结果；本阶段不安装、不修改源码，也不将 `accelerate --num_processes 1` 当成自动绕过DeepSpeed。

**Confirmed** — 官方安装顺序是先创建 `starvla-libero`，再用 `--no-deps` 安装LIBERO/OpenVLA-OFT/StarVLA，保护既定pins（`ENVIRONMENTS.md:25-43`）。LIBERO配置由 `scripts/setup/configure_libero.py` 写入机器本地 `.local/libero/config.yaml`。这是未来安装时的写操作，本次没有执行。

**Confirmed** — `robotwin` 的setuptools上限 `<81` 是因为SAPIEN仍导入pkg_resources（yml:7-8）。`_install.sh` 不只是pip requirements：会patch已安装SAPIEN/MPLib文件（:19-60）、clone并可编辑安装CuRobo v0.7.8、设置编译器/单worker并安装warp-lang（:63-84）。因此不应在只读审计中调用。最终仍需核对pip实际解算结果是否保持yml约束。

**Confirmed** — RoboTwin assets不在Git，文档估计约30GB（`ENVIRONMENTS.md:66-72`）。`scripts/setup/setup_robotwin_assets.sh:6-8,10-43` 要求三组 `background_texture/embodiments/objects`；可链接已有完整目录或下载并解压。`:45-51` 检查objaverse列表及aloha-agilex embodiment。安装脚本同样不应在本阶段执行。

**Confirmed** — FastWAM使用独立Torch2.7.1/CUDA12.8环境；OpenVLA-OFT standalone与GR00T也有可选环境，GR00T以其uv.lock为准（`ENVIRONMENTS.md:74-119`）。本阶段优先PI0/PI0.5/Qwen3-OFT，无需为了“完整”安装全部可选栈。

## D2. 子模块与真实版本边界

**Confirmed** — `SUBMODULES.md:3-14` 与 `.gitmodules:1-26` 表明项目改版子模块以 **同一个RawVLA仓库的 `submodules/*` 分支**托管，并非跟踪StarVLA/RoboTwin上游最新main。根Git gitlink SHA才是本次审计版本。Agent C读取的实际checkout：

```text
57d508cb180d0510ac9060358009c0f6ffaac420 starVLA
8e9e66544e2e9efc5910721b8d97d6dee806fa20 third_party/RoboTwin
```

其余子模块gitlink、实际HEAD、sparse checkout和工作区状态以 `REPO_SNAPSHOT.md` 为准；root后续已初始化全部六个直接gitlink，Isaac-GR00T四个嵌套子模块未初始化，不可将gitlink记录描述为已读取嵌套依赖源码。starVLA在此版本承担RAW frontend registry、NPZ dataset、训练器、policy serving、LIBERO/RoboTwin adapters；根库 `baselines/` 中的模型实现由该集成调用。证据：`SUBMODULES.md:9-14` 及 `starVLA/examples/simBenchmarks/Robotwin/eval_files/model2robotwin_interface.py`。

**Confirmed** — `SUBMODULES.md:18-25` 提醒gitlink所指commit应可匿名fetch；本次starVLA/RoboTwin已成功初始化到上述实际SHA。该结果不自动证明所有可选子模块、其递归依赖和后续大型assets都可获取。

## D3. 模型清单、路径、来源与体量

以下是2026-11-16只读元数据结果。GB为十进制；精确bytes列优先。`public/ungated` 只表示查询时访问无需门禁，不等于许可证完全明确。训练权重文件还需要与其配套的config、statistics、tokenizer或processor，不能只拿一个同名权重文件。

| 用途 | 来源、查询时revision | 必要权重体量 | 配置/本地目标 | 证据等级 |
|---|---|---:|---|---|
| LIBERO PI0 bf16 | [`tenstep/pi_model_starvla`](https://huggingface.co/tenstep/pi_model_starvla/tree/13b0c90e378bf7a873bf6cea99c68e6aef04f5c8), `13b0c90e378bf7a873bf6cea99c68e6aef04f5c8` | `8,058,327,112` bytes（8.06GB） | `$RAWVLA_ROOT/openpi_converted_protocol/pi0_libero_starvla/bfloat16/model.safetensors` | Confirmed（公共文件与config路径）；原始转换等价性Partially Confirmed |
| LIBERO PI0.5 bf16 | 同repo、同revision | `8,522,039,320` bytes（8.52GB） | `.../pi05_libero_starvla/bfloat16/model.safetensors` | 同上 |
| LIBERO Qwen3-OFT | [`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/tree/39177146f142f8805d86fcc29d2a17c2228e6073), `39177146f142f8805d86fcc29d2a17c2228e6073` | checkpoint `9,785,061,050` bytes（9.79GB） | `starVLA/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt` | Confirmed |
| Qwen3 base VLM | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17), `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 两safetensors分别 `4,967,229,296`、`3,908,490,048`；整个repo `8,887,292,732` bytes | `starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` | Confirmed |
| RoboTwin PI0 原始subset | [`Heisen0928/pi0_robotwin`](https://huggingface.co/Heisen0928/pi0_robotwin/tree/f163ea4a7d4b4806a811644117e2e4e6ab29e1a9), 固定 `f163ea4a7d4b4806a811644117e2e4e6ab29e1a9` | 只含45000 metadata/params/assets的25文件共 `12,015,690,989` bytes（12.02GB） | `$RAWVLA_MODEL_ROOT/pi0_robotwin/45000/...`，JAX/Orbax | Confirmed |
| RoboTwin PI0 转换后 | 上一项经兼容OpenPI converter | **Unknown** 精确产物字节数；另预留约10GB量级及转换内存（估计） | `$RAWVLA_MODEL_ROOT/pi0_robotwin_pytorch_45000/model.safetensors` | Partially Confirmed（转换入口存在，未转换） |
| RoboTwin PI0.5 | [`SidneyXie/pi05_robotwin`](https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416), 固定 `e49e2ab6c11f07511573b67261bd129e88d0a416` | model `9,354,050,752`；完整9文件repo `9,354,082,616` bytes（9.35GB） | `$RAWVLA_MODEL_ROOT/pi05_robotwin/model.safetensors` | Confirmed |
| README RoboTwin Qwen模型1 | [`StarVLA/Qwen3-VL-OFT-RoboTwin2-All`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All/tree/9fae49581755f57944ec57f6787d214c0c0143f5), `9fae49581755f57944ec57f6787d214c0c0143f5` | `steps_140000_pytorch_model.pt`: `9,785,133,528` bytes | `$CKPT_ROOT/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/...` | Confirmed |
| README RoboTwin Qwen模型2 | [`StarVLA/Qwen3-VL-OFT-Robotwin2`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2/tree/e9cdfc38a0db968bf3a865bcb542d3d455952226), `e9cdfc38a0db968bf3a865bcb542d3d455952226` | `steps_40000_pytorch_model.pt`: `9,785,132,794` bytes | `$CKPT_ROOT/Qwen3-VL-OFT-Robotwin2/checkpoints/...` | Confirmed |

大小验证方法（只访问JSON元数据，不下载safetensors/pt）：

```python
import json, urllib.request
repo='SidneyXie/pi05_robotwin'
rev='e49e2ab6c11f07511573b67261bd129e88d0a416'
d=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true'))
print(d['sha'],d.get('gated'),d.get('cardData',{}).get('license'))
for f in d['siblings']:
    print(f['rfilename'], f.get('size'))
```

API输出已核对，未调用snapshot_download/hf download。本次查询均`gated=false`。pi0 subset统计严格匹配 `download_pi0.sh:25` 的include patterns，不能把整个源repo（可能有其他训练状态）计成最低下载量。

## D4. 获取方法及容易混淆的对象

### LIBERO

**Confirmed** — `tools/checkpoints/download_starvla_libero.py:27-47,128-169` 默认下载 `StarVLA/Qwen3-VL-PI-LIBERO-4in1`（Qwen3-PI）及其Qwen3 base，**它不是PI0或PI0.5**。`--all` 列表也未包括OpenPI PI0/PI0.5。要取本次主表Qwen3-OFT，应明确指定：

```bash
# 下一阶段才运行；会下载约18.67GB的policy+base及元数据
python tools/checkpoints/download_starvla_libero.py \
  --repo-id StarVLA/Qwen3-VL-OFT-LIBERO-4in1
```

脚本依赖`huggingface_hub`、`huggingface-cli`与`aria2c`（:20,89-106,116-124）；environment-libero.yml未声明aria2c。脚本未给HF revision，后续复现应改用**独立下载命令/清单**显式固定上表revision并保存SHA256，不能仅记录repo名称。`validate_starvla_libero.py` 默认遍历全部五种checkpoint（:23-29,61-64），只准备一个时不能把其他文件缺失当作已准备模型失败；可逐项检查config/statistics/权重。`--load` 会实例化大模型，不属于metadata check。

**Confirmed** — LIBERO PI0/PI0.5的公开预转换源被starVLA文档点名：`starVLA/examples/simBenchmarks/LIBERO/train_files/openpi/README.md:6-7,33-35`。HF API确认含与root config匹配的 `pi{0,05}_libero_starvla/bfloat16/{model.safetensors,assets/physical-intelligence/libero/norm_stats.json}`。无需为了本项目最小路径先自行训练PI policy；不要下载无关的base/fp32两套变体。

下一阶段精确下载模板（本阶段未执行）：

```bash
hf download tenstep/pi_model_starvla \
  --revision 13b0c90e378bf7a873bf6cea99c68e6aef04f5c8 \
  --include 'pi0_libero_starvla/bfloat16/**' 'pi0_libero_starvla/config.yaml' 'pi0_libero_starvla/dataset_statistics.json' \
  --local-dir "$RAWVLA_ROOT/openpi_converted_protocol"
```

PI0.5把上面的`pi0_libero_starvla`换成`pi05_libero_starvla`。配置路径证据：`configs/libero/pi0.yaml:10,22,29`；`pi05.yaml:10,22,29`。

**Partially Confirmed（来源链）** — HF `conversion_metadata.json` 仅记作者绝对本地source path（pi0_libero/pi05_libero）、variant、format，未提供OpenPI source revision/源权重hash；不能证明该转换与作者论文实际权重逐bit一致。`pi0_05_to_starvla.py` 的转换入口可读（:200-225），但本阶段未运行。该转换器将assets复制到output root（:84-91,214），而root训练配置选variant下的assets；如果自行转换，需要显式核对最终布局，不能盲目照搬路径。上表公共repo现有variant/assets已存在。

**Confirmed（必要tokenizer）** — 两个PI配置另需本地 `paligemma_tokenizer.model`；`starVLA/.../PI0.py:98-115 LazyPaliGemmaTokenizer._load` 不自动联网下载。tenstep repo根目录只含.gitattributes/README，无tokenizer。官方 [OpenPI tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py) 的 `PaligemmaTokenizer.__init__` 指向 `gs://big_vision/paligemma_tokenizer.model`。等价HTTPS URL的只读HEAD：

```text
https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
HTTP 200; Content-Length 4264023
Last-Modified Wed, 27 Mar 2024 13:53:25 GMT
```

未来下载后放 `$RAWVLA_ROOT/openpi_converted_protocol/paligemma_tokenizer.model`（LIBERO）或设置 `RAWVLA_TOKENIZER_PATH`（RoboTwin base config:31）。**Unknown** tokenizer独立许可与需接受的上游模型条款，本阶段HEAD不确认此法律条件；应按提供者当前条款核实，不应从Apache2模型卡推断所有上游资产许可。

### RoboTwin PI0 / PI0.5

**Confirmed** — `experiments/simulation/robotwin/backbones/README.md:9-16,39-103` 和 `download_pi0.sh:18-31` 固定PI0为full finetune的45000 Orbax checkpoint，不可当LoRA；需兼容OpenPI checkout，含 `examples/convert_jax_model_to_pytorch.py` 和注册项 `pi0_base_aloha_robotwin_full`。`convert_pi0_jax_to_pytorch.sh` 调用官方接口，bf16输出；**Unknown** 合适OpenPI checkout完整SHA，因为本项目未提供固定OpenPI子模块或该转换环境lock。不要从任意当前上游main复制一个converter就当已复现。

**Confirmed** — PI0统计必须保持原 `pi0_robotwin/45000/assets/robotwin_50_clean/norm_stats.json`，权重与stats分别设置（guide:98-106,153-159；root pi0.yaml:7,16）。作者记录stats SHA256为 `7bd3ec58acee3fe73bc6423ab12b904821b05e98d69a68c8a89cc1916208f64b`（guide:181），本阶段未下载所以未本地验hash。

**Confirmed** — PI0.5已是PyTorch/LeRobot，不需JAX转换；必须保留pre/postprocessor JSON及normalizer safetensors，后续运行 `export_pi05_policy_norm_stats.py --checkpoint-dir ...`，将action/state统计等价导出为 `policy_norm_stats.json`，不是从RAW数据重新估计（guide:108-147）。作者记录导出JSON SHA256为 `229a4a2d6cfd1ba4eb2bc0b3aa57f049094032939130e4901485e9ea9910f5f7`（:182），本阶段未验本地产物。下载脚本本身只检查文件存在（download_pi05.sh:27-37），不保证checksum或模型行为。

未来命令（会下载大权重，本阶段未执行）：

```bash
export RAWVLA_MODEL_ROOT=/absolute/path/to/robotwin2_backbones
bash experiments/simulation/robotwin/backbones/download_pi0.sh "$RAWVLA_MODEL_ROOT/pi0_robotwin"
bash experiments/simulation/robotwin/backbones/download_pi05.sh "$RAWVLA_MODEL_ROOT/pi05_robotwin"
python experiments/simulation/robotwin/backbones/export_pi05_policy_norm_stats.py \
  --checkpoint-dir "$RAWVLA_MODEL_ROOT/pi05_robotwin"
```

PI0的转换还需设置OPENPI_ROOT/OPENPI_PYTHON（guide:63-88）。完整具体转换命令保留官方guide，待兼容版本确认后运行。

**Unknown / 需后续解决** — 上述都是冻结policy，不等于训练好的RawVLA frontend。guide:171-172仅给 `steps_100_raw_frontend_pytorch_model.pt` 例子，未提供所有主表frontend checkpoint的固定公开下载清单。可以从随机初始化训练frontend，但“公开base policy可取”不能直接推出“作者所有结果可直接评估”。

## D5. 许可证范围

| 对象 | 本次核实的许可/门禁 | 证据等级与限制 |
|---|---|---|
| 根代码、starVLA、RoboTwin | 各自LICENSE标MIT | Confirmed，`LICENSE:1-13`、`starVLA/LICENSE:1-16`、`third_party/RoboTwin/LICENSE:1-13`；保留各自归属/声明 |
| `tenstep/pi_model_starvla` | 模型卡`license: apache-2.0`，ungated | Confirmed（卡声明）；原始source权重版本链Partially Confirmed |
| `SidneyXie/pi05_robotwin` | 模型卡Apache2，ungated | Confirmed（卡声明） |
| `Qwen/Qwen3-VL-4B-Instruct` | 模型卡Apache2，ungated | Confirmed（卡声明） |
| `Heisen0928/pi0_robotwin` | API cardData无license，ungated | Unknown 权重许可；不能用根代码MIT替代 |
| StarVLA Qwen3-OFT LIBERO及两款RoboTwin finetunes | API cardData无license，ungated | Unknown finetuned权重独立条款；基础Qwen的Apache2不自动补齐此声明 |
| PaliGemma tokenizer | 可匿名HEAD，独立许可未查明 | Partially Confirmed 可访问；Unknown完整许可 |

这些是源码和元数据事实，不是新增审批流程。后续使用者应保存所选模型版本的model card/LICENSE，缺失时向权重提供者查明适用条款。仿真assets与上游数据可能另有许可，不应套用根仓库MIT。

## D6. 作者硬件与资源建议

**Confirmed（论文声明）** — [论文 §5 Implementation Details 与 Appendix D.1](https://arxiv.org/html/2609.37530v1) 明确每个冻结policy对应ISP训练使用 **一张 NVIDIA A100 80GB**，batch1、梯度累积8，预算2000 optimizer updates。不能将starVLA文档里的8 GPU整policy训练当成RawVLA frontend训练硬件要求。`starVLA/.../LIBERO/README.md:9`提及A100/RTX4090是上游工作流验证；`experiments/isp_perturbation/configs/libero_zeroshot_profiles.yaml`里的H20是特定脚本profile，并不能覆盖本项目全部主实验。

**Confirmed（另一个实验阶段）** — [论文 Appendix H.3](https://arxiv.org/html/2609.37530v1#A8.SS3) 的真实机器人PI0.5全参数微调使用 **8张NVIDIA B300**，40,000 updates、global batch256；随后冻结policy，另训练ISP 2,000步。这与主仿真benchmark的冻结policy前端训练不同，不是本阶段最小复现必须准备的硬件。

**Confirmed（真实机器人平台）** — [Appendix H.1/H.2](https://arxiv.org/html/2609.37530v1#A8.SS1) 给出两台ROKAE AR5机械臂、Songling夹爪、三台同步HIKROBOT MV-CS020-10UC工业相机；agent镜头4mm f/2.8、腕部3.5mm f/4。采集为12-bit Bayer RAW、30FPS，经黑/白电平和demosaic成为三通道linear RAW；这些物理硬件不需要用于LIBERO/RoboTwin最小仿真复现。

**Unknown** — 作者主实验CPU型号/内存、磁盘、OS完整镜像、确切驱动版本、每种policy推理峰值显存和wall-clock未由本次证据充分建立；不把推荐值写成作者实际配置。

建议按阶段准备，均为规划估计：静态/JSON/NPZ metadata用普通CPU笔记本即可；单帧frontend可用CPU无需policy权重；VLA前向/梯度最稳妥准备A10080GB及对应一套policy资产；仿真再加NVIDIA图形驱动、Vulkan和约30GB assets。Qwen3单policy+base约18.67GB下载（不含环境），PI bf16约8–9.35GB级；Orbax PI0需源权重+转换后副本+转换缓存。实际磁盘预留应显著高于这些文件和，不能按下载字节直接估计环境总占用。

## D7. 下一阶段验收顺序

1. **Confirmed（现有工具）** — 在目标环境运行 `python scripts/setup/check_environment.py rawvla` 会调用完整frontend smoke（脚本:91-94,144-145）；这已是计算测试，并非只检查imports，本阶段没有执行。
2. **Confirmed（现有工具）** — LIBERO环境执行 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/setup/check_environment.py starvla-libero --render`；RoboTwin用 `VK_ICD_FILENAMES=... python scripts/setup/check_environment.py robotwin --render`（:97-129）。该RoboTwin check只验证assets和renderer创建，不证明一个task episode或policy server可运行。
3. **Partially Confirmed（计划）** — 仅准备所选policy/config/stats/tokenizer，验证哈希、路径和normalization；再做单batch/梯度；最后做评估报告中的单episode。报告中的运行命令均为下一阶段路线，不代表本次实际执行过。
