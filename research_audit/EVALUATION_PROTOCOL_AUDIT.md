# RawVLA Phase 1 — RoboTwin 评估协议审计

审计日期：2026-11-16（Asia/Shanghai）。独立 Agent C；仅静态阅读、JSON 统计和公开元数据查询，未启动服务器、仿真或模型。根仓库 `ba9c56b04da2c4931b70d8844710f5928d43775b`；starVLA `57d508cb180d0510ac9060358009c0f6ffaac420`；RoboTwin `8e9e66544e2e9efc5910721b8d97d6dee806fa20`。下文路径均相对于根仓库。Confirmed 指源码/输出直接证明；Partially Confirmed 指接口静态成立但尚未运行；Unknown 指缺少运行结果或作者材料。

## C1. 冻结的 3,250 案例

**Confirmed** — 位置是 `experiments/simulation/robotwin/manifests/robotwin2_test_manifest_50_seeds_3250_rollouts.json`，不是 Hugging Face 中的测试轨迹。`README.md:149-164` 明确训练数据与测试 manifest 分离。论文 [Appendix B.2](https://arxiv.org/html/2609.37530v1#A2.SS2) 描述 13×50×5；正文 §5 使用 PI0、PI0.5 进行 RoboTwin 主表比较。

**Confirmed** — manifest 顶层是 JSON object，包含 `benchmark/version/benchmark_seed`、`tasks/tasks_count`、`base_seeds_per_task`、`dataset`、`entries`、`lighting_domain_configs/ev_ranges` 和 `raw_sensor`。`entries` 从第 691 行起，首条第 692-714 行可作字段定义实例：

```json
{
  "task_name": "grab_roller", "task_id": 0,
  "dataset_seed": 0, "base_seed": 0, "base_seed_id": 0,
  "dataset_episode_index": 0, "episode_idx": 0,
  "dataset_relative_dir": "grab_roller/aloha-agilex_clean_50",
  "lighting_domain": "ExtremeLow",
  "lighting_ev": -8.848987530150817,
  "lighting_sample_unit": 0.30202493969836497,
  "lighting_scale": 0.002168649278941593,
  "lighting_strategy": "environment_only",
  "extra_light_intensity": 0.0, "sensor_saturation_ev": 0.0,
  "noise_seed": 1386085259, "raw_white_level": 3.5,
  "raw_sensor_noise_enabled": true,
  "raw_sensor_shot_noise": 0.000025,
  "raw_sensor_read_noise": 0.0000000390625
}
```

**Confirmed** — 只读 Python 统计输出：

```text
N 3250 unique(task_name,base_seed_id,lighting_domain) 3250
domains {'ExtremeLow':650,'Low':650,'Normal':650,'Over':650,'ExtremeOver':650}
all_base_seed_equals_dataset_seed True
```

13 个任务是 `grab_roller, handover_mic, lift_pot, move_can_pot, open_laptop, pick_dual_bottles, place_object_basket, place_dual_shoes, place_phone_stand, put_bottles_dustbin, put_object_cabinet, stack_blocks_two, stack_bowls_two`（manifest:75545-75560）。每任务各有 50 个实际 dataset seed；不能简单替换为整数 0…49。例如 `grab_roller` 的 dataset episode 21 对应实际 seed 22，`base_seed_id=21`；`episode_idx=105` 是展开案例索引，并非实际仿真 seed。

**Confirmed** — 五域 EV 范围分别为 `[-9,-8.5],[-7,-6],[-0.5,0.5],[1,1.5],[2,2.5]`；所有条目 `lighting_strategy=environment_only`、white level 3.5、shot/read noise 如上、额外灯与 saturation EV 都为 0。证据：manifest:75443-75544；逐项 `all(entry[key]==constant for entry in entries)` 结果均 True。`raw_sensor` 声明三通道 linear float32、[0,1]、非量化，而不是 Bayer mosaic。

## C2. 当前 runner 实际执行什么

**Confirmed** — `scripts/evaluation/run_robotwin2_starvla_rgb_eval.sh` 没有读取上述 manifest。静态链路：

```text
root README example
→ run_robotwin2_starvla_rgb_eval.sh:614 main
→ resolve_tasks:345 / checkpoint-mode-task loops:651-688
→ run_eval_task:369
→ starVLA/.../Robotwin/eval_files/eval.sh:94-102
→ third_party/RoboTwin/script/eval_policy.py:65 main / :190 eval_policy
→ model2robotwin_interface.py:412 eval
→ ModelClient.step / WebSocket server / simulator take_action
```

`rg -n 'manifest|json.load|json.loads'` 对外层脚本及 `eval_policy.py` 无命中；`eval_policy.py:79-80` 读取的是 task YAML，`:490-508` 读取 deployment YAML，而非 frozen rollout JSON。README:158-163 也直接声明此差异。

**Confirmed** — 根 README:214-216 的 `one adjust_bottle` 表示“给定 task list”，不是一次 episode。默认 `TRIALS=50`（脚本:17）、两个 checkpoint（:88-92）、easy/hard（:85-87），故成功完成时为 **2×2×1×50=200 次 policy rollouts**；还可能先执行 expert/planner 检查。`adjust_bottle` 不在 frozen 13 tasks 中。`full` 是其硬编码 50 个任务（:32-83），不是论文 manifest 的 full；默认矩阵 2×2×50×50=10,000。

**Confirmed** — easy 使用 `demo_clean_rgb_eval`；hard 使用 `demo_randomized_no_light_rgb_eval`。`write_runtime_task_configs:313-343` 从 simulator YAML 生成配置，关闭录像；hard 禁用随机灯光，但仍保留其他 domain randomization；两者加 white level 3.5 和 noise。默认没有设置五个 `ROBOTWIN_LIGHTING_DOMAIN`。因此 easy/hard 不能替代五种 illumination regimes。

**Confirmed** — 默认输入键名是 `rgb`：`model2robotwin_interface.py:417` 的 `ROBOTWIN_IMAGE_OBS_KEY` 默认如此，按 head/left/right 顺序取三个视角（:418-423）。但相机 `third_party/RoboTwin/envs/camera/camera.py:384-391` 明确 `rgb=raw_view8`（uint8 RAW可视化），`render_rgb=renderer RGB`，`raw_linear`才是未量化float RAW。外层 runner没有设置输入键，也没有默认加载训练好的RAW frontend；因此它的`rgb`文件名不证明使用真正renderer RGB，更不能等同论文主表。真正RAW frontend transport由server metadata激活（adapter:77-84、130-140），还必须显式选择正确输入键及checkpoint。

**Confirmed** — seed 不是第四个 shell 参数的字面值：`eval.sh:101` 把它传为 `--seed`，`eval_policy.py:161` 默认转成 `100000*(1+seed)`；用 `ROBOTWIN_EVAL_START_SEED` 可覆盖。显式 `ROBOTWIN_EVAL_SEEDS` 在 `eval_policy:240-243,302-304` 才是实际 seed 列表。`ROBOTWIN_TEST_NUM`（:163）控制 policy 测试计数；simulator YAML 的 `episode_num:50` 不承担此作用。

**Confirmed** — 默认会先运行 expert `play_once` 生成指令/检查可行性（`eval_policy.py:219,311-343`）。`ROBOTWIN_FORCE_EVAL_SEED=1` 仍不绕过前面的 expert exception。若显式 seed 列表遇到 expert exception，`now_id` 没增加，下一轮可能重用相同 seed。`ROBOTWIN_USE_DATASET_INSTRUCTIONS=1` 才绕过 expert check，但要求相应单个 `instructions/episodeN.json`，由 `ROBOTWIN_POLICY_AUDIT_DATA_ROOT` 与 `START_EPISODE` 定位（:210-219,356-373）。`START_EPISODE` 是 dataset episode index，不能把 seed22误填作episode21。

**Confirmed（静态执行障碍）** — 外层脚本:608、690 直接执行 `scripts/evaluation/summarize_robotwin2_eval.py`，而 `git ls-files -s` 输出 mode `100644`，本次 checkout 无可执行位。正常 POSIX 下直接执行会受权限阻碍；`:608` 的错误被 `|| true` 忽略，`:690` 没有。未通过 chmod 改动官方文件。本阶段不据此推断模型本身不可运行。后续可显式 `python scripts/evaluation/summarize_robotwin2_eval.py ...`，或采用下述直连测试。该脚本只统计 DONE/FAILED marker（summary.py:31-43），不是 episode 成功率汇总器。

## C3. 最小自定义单任务、单 policy episode

**Partially Confirmed** — 以下是依据已核实参数编写的下一阶段命令，本阶段没有运行。建议绕过外层矩阵脚本，用已知的 Qwen3 RoboTwin RGB policy 验证 server→client→simulator。它属于自定义 smoke test，不宣称论文复现。

前提：两个独立 Python 环境、匹配的 GPU 驱动/Vulkan、已准备 RoboTwin assets；一个 Qwen3 RoboTwin finetuned checkpoint 及其同级 run `config.yaml`、`dataset_statistics.json`；Qwen3 base model。`eval.sh` 只接受 checkpoint 路径还不代表只需一个 `.pt`，adapter:204-211 以及 server 的 configuration/statistics 读取都依赖附属文件。

终端 A，从根仓库执行（替换绝对路径；此命令会真正加载模型）：

```bash
export RAWVLA_ROOT=/absolute/path/to/RawVLA
export QWEN3_CKPT=/absolute/path/to/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
export QWEN3_BASE=/absolute/path/to/Qwen3-VL-4B-Instruct
cd "$RAWVLA_ROOT/starVLA"
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 /absolute/path/to/starvla-libero/bin/python \
  deployment/model_server/server_policy.py \
  --ckpt_path "$QWEN3_CKPT" --port 5694 --use_bf16 --idle_timeout -1 \
  --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE"
```

终端 B，从根仓库执行；确认 A 已打印 server ready 后运行：

```bash
export RAWVLA_ROOT=/absolute/path/to/RawVLA
export QWEN3_CKPT=/absolute/path/to/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
cd "$RAWVLA_ROOT"
ROBOTWIN_PATH="$RAWVLA_ROOT/third_party/RoboTwin" \
ROBOTWIN_PYTHON=/absolute/path/to/robotwin/bin/python \
ROBOTWIN_TEST_NUM=1 ROBOTWIN_EVAL_SEEDS=0 ROBOTWIN_IMAGE_OBS_KEY=render_rgb \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
bash starVLA/examples/simBenchmarks/Robotwin/eval_files/eval.sh \
  grab_roller demo_clean qwen3_rgb_smoke 0 0 "$QWEN3_CKPT" 5694 127.0.0.1
```

验收：client 接通；输出一个 policy rollout 的 `Success!` 或 `Fail!`、`.../1` 与实际 seed；有 `third_party/RoboTwin/eval_result/.../_result.txt`；动作来自模型、14D、仿真运行至成功或 step limit。模型成功率不作为单次 smoke 的通过条件。默认可能额外执行 expert 检查，且原 `demo_clean` 开启视频，需 FFmpeg。

若要**严格固定一个实际 seed 且避免 expert rollout**，在 B 的命令环境中再设置：

```bash
ROBOTWIN_USE_DATASET_INSTRUCTIONS=1
ROBOTWIN_POLICY_AUDIT_DATA_ROOT=/absolute/path/to/robotwin2/dataset
START_EPISODE=0
```

须将这些变量 export 或作为同一条命令的环境赋值；需要 `grab_roller/aloha-agilex_clean_50/instructions/episode0.json`，含 `unseen` 非空列表（deployment YAML:6）。这个分支不需要整套 HDF5，也不应设置会触发参考图像读取的 `ROBOTWIN_POLICY_AUDIT_DIR`。没有该指令文件时，用普通 smoke 分支并如实记录 expert 前置步骤。服务器测试结束后在终端 A 用 Ctrl-C 停止。

如果使用外层 runner 缩小矩阵，必要参数是 `ROBOTWIN_CUSTOM_CKPT_NAME/PATH`、`MODES_CSV=easy`、`TRIALS=1`、`one grab_roller`；`CKPTS_CSV` 过滤本身不足以免除另外 checkpoint 的存在性检查（:634-637），还要处理上述汇总脚本执行权限。它会在 simulator task_config 下生成运行 YAML，因此不在本次只读审计中执行。

资源估计：CPU 8 核、RAM 32–64 GB、单个 CUDA GPU 24–48 GB 可作为推理尝试起点，但**Unknown** 最低显存，双进程共用显存需实测；更稳妥用 A100 80GB。assets 文档约30GB，Qwen policy+base下载体量约18.67GB（见环境报告），环境/缓存额外预留。单次可能数分钟，首次 CuRobo 编译更久；均为计划值，不是本阶段测量。

## C4. 正式论文协议还需要什么

**Confirmed** — 已存在关键底层控制：`third_party/RoboTwin/envs/lighting.py:106-159 apply_lighting`；`:181-196 apply_lighting_from_env`；`envs/_base_task.py:124` 在 setup 时调用。环境变量支持 `ROBOTWIN_LIGHTING_DOMAIN`、`ROBOTWIN_LIGHTING_STRATEGY`、`ROBOTWIN_LIGHTING_SAMPLE_UNIT`。`paired` 会按 task+实际seed+domain+benchmark_seed计算 EV 与 noise seed（:162-178），与首条 manifest noise seed 只读复算一致。正式执行最好逐条使用 manifest 显式值并对全部3250条验证，而不是只验证首条或依赖默认 `full/0.5`。

**Confirmed（Default ISP来源需单独核验）** — HDR分支 `camera.py:352-368` 给 `raw_linear` 加噪并按white level归一化，但将 `default_isp` 直接设为renderer的 `render_rgb_float`；它并非在这段代码里由加噪后的同一份 `raw_linear` 重处理得到。选择 `default_isp` 只能确定读取哪个产物，不能自动证明与论文固定ISP基线在噪声、white level和映射上完全一致。正式adapter应将这条来源差异与论文Appendix A/B.2核对，保留实际相机metadata；本次没有逐像素或闭环验证，不将其定性为已证实的实验Bug。

**Partially Confirmed** — 下一阶段在独立 adapter/launcher 中逐条调度可形成完整协议；当前不存在可直接宣称执行这份 JSON 的一键命令。最小实现合同：

1. 固定本报告根/子模块 SHA、policy revision、frontend checkpoint SHA256；读取 frozen manifest，唯一键使用 `(task_name,base_seed_id,lighting_domain)`。
2. 用 `dataset_seed` 创建场景，用 `dataset_episode_index` 定位指令；所有 ISP 方法和两款 policy 使用同一批实际 seeds/指令，拒绝静默替换或跳过。
3. 以 `demo_clean` 为场景基础，应用每条 `environment_only`、`lighting_sample_unit`、white level、shot/read noise和`noise_seed`。若沿用 env hook，必须显式设置domain/strategy/sample，并验证实际metadata与条目；`paired`同时重写noise seed，需核验全部条目。
4. RawVLA/神经ISP输入显式选 `ROBOTWIN_IMAGE_OBS_KEY=raw_linear`；该键定义在camera.py:387，`raw_float32`并非simulator observation键，不能把NPZ字段名搬到这里。fixed Default ISP应选择并核验对应ISP输出`default_isp`，renderer RGB smoke选`render_rgb`；uint8 `rgb`不满足frontend的float32断言（adapter:267-275）。按policy checkpoint匹配动作/状态normalization、相机顺序、action horizon、PI0 delta/PI0.5 state token等。
5. 每 episode 重置客户端历史与 frontend recurrence（adapter.reset:91-102），输出成功/失败、实际seed、完整光照/噪声、policy/frontend身份与错误；错误列为错误，不可据DONE marker算成功。
6. 先跑一个manifest entry验证，再扩为单任务50×5=250；最后每个policy×方法完成3250条，按五域各650条报告成功率，并核对唯一键覆盖、重跑去重、无缺失。论文主表用 PI0/PI0.5，不用 README 的两个Qwen模型替代。

**Confirmed（frontend启用接口）** — `server_policy.py:77-83` 接受重复`--config_override KEY=VALUE`；构建前合入checkpoint config（`base_framework.py:272-279`）。`registry.py:591-644 build_raw_frontend`读取`framework.raw_frontend.name`、`checkpoint`并load frontend state。用于独立frontend的override形状如下，必须连同该frontend训练时完整架构配置一起核对：

```text
--config_override framework.raw_frontend.name=rawvla
--config_override framework.raw_frontend.checkpoint=/absolute/path/to/steps_2000_raw_frontend_pytorch_model.pt
--config_override framework.raw_frontend.source_key=raw_burst_float32
--config_override framework.raw_frontend.output_contract=rgb_chw_float32_0_1
--config_override framework.raw_frontend.burst_frames=6
--config_override framework.raw_frontend.rnn_frames=6
--config_override framework.raw_frontend.trainable=false
```

`PolicyServerWrapper.metadata:175-192`发送name/source_key/burst/rnn；client按metadata把`raw_linear`历史打包到`raw_burst_float32`（adapter:130-140,279-294）。没有checkpoint时registry会随机初始化frontend，因此“name=rawvla且成功启动”不证明加载了训练结果。

**Partially Confirmed（PI静态模型接口）** — PI0构造frontend（PI0.py:218-231），`predict_action:518-532`也调用frontend；PI05继承PI0（PI05.py:52-58），所以模型层提供该路径。但generic `PolicyServerWrapper:124`直接读取`framework.action_model`，`read_mode_config:369-395`要求特定run/checkpoints布局的config.yaml及dataset_statistics.json；原RoboTwin PI权重包不能仅凭路径直接喂给此server。client `normalize_state:225-242`和默认deployment的min_max/abs、14D动作约定也需要按PI训练配置适配。**Unknown** 从现有下载的PI权重→generic RoboTwin server/client→闭环动作的完整运行可用性；本报告不提供声称已验证的PI一键评估命令。

**Unknown / 阻碍论文数值复现** — 本阶段未确认作者全部最终frontend权重的公开下载定位、原始每episode结果和精确运行命令。根LIBERO runner的baseline默认文件名存在4k/8k变体（`experiments/simulation/libero/scripts/run_libero_models.sh:178-187`），不能以默认文件名证明论文2k预算；`RAW_FRONTEND_CHECKPOINT`覆盖仅在rawvla分支使用（:167）。需检查每个方法实际加载checkpoint身份、文件存在性与2k训练provenance。可训练配置及base policy来源的存在，不等于这组评估集成已经运行通过。需要Evaluation Agent先实现并审核manifest驱动及PI适配器，再由Reproduction Agent做小规模验证。

## C5. 无仿真的协议验收命令

```bash
python - <<'PY'
import collections, json
p='experiments/simulation/robotwin/manifests/robotwin2_test_manifest_50_seeds_3250_rollouts.json'
d=json.load(open(p)); e=d['entries']
assert len(e)==3250
assert len({(r['task_name'],r['base_seed_id'],r['lighting_domain']) for r in e})==3250
assert set(collections.Counter(r['lighting_domain'] for r in e).values())=={650}
assert len({r['task_name'] for r in e})==13
assert all(r['base_seed']==r['dataset_seed'] for r in e)
print('manifest structural checks PASS')
PY
```

这只验证结构和覆盖数量，不证明 simulator 实际执行了这些案例。
