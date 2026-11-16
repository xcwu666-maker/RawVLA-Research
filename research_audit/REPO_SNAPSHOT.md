# RawVLA Phase 1：仓库快照与审计边界

审计日期：2026-11-16（按客户端日期，Asia/Shanghai）。本报告对应本次读取的固定版本，不把后续 `main` 当成相同版本。

## 1. 快照结论

| 项目 | 记录 | 证据等级 |
| --- | --- | --- |
| 官方仓库 | https://github.com/ShuhongLL/RawVLA | Confirmed |
| 本地只读审计 checkout | `/workspace/work/RawVLA`，审计报告在 `/workspace/research_audit/` | Confirmed：本次 shell 输出 |
| 根仓库完整 SHA | `ba9c56b04da2c4931b70d8844710f5928d43775b` | Confirmed：`git rev-parse HEAD` |
| 当前分支 | `main` | Confirmed：`git branch --show-current` |
| 远端 main | 与上述 SHA 相同 | Confirmed：`git ls-remote --heads --tags origin` |
| commit 时间 | `2026-10-04 08:04:32 +0000` | Confirmed：`git log -1 --format=fuller` |
| commit 说明 | `Separate optional ISP and real-robot README sections` | Confirmed：同上 |
| 根仓库工作区 | 干净；审计前后均无源码差异 | Confirmed：`git status --porcelain=v1 --untracked-files=all` 与 `git diff --stat` 均无输出 |
| 六个直接子模块 | 实际 HEAD 均与 gitlink 一致，工作区干净 | Confirmed：逐个 `git -C <path> rev-parse HEAD` / `status --porcelain`，见下表 |
| 数据、权重、仿真 | 未下载数据集或预训练权重；未训练、未执行模型前向或仿真；未提交或推送 | Confirmed：本次操作记录 |

这里的只读指不修改官方版本化源码。为了能调查源码，创建了全新浅克隆，初始化了子模块及本地 Git 配置；这些操作不写入上游，也不改变版本化文件。论文 HTML、网页/API 元数据、临时分析材料在 `work/`。本阶段未安装训练依赖。

## 2. 子模块：锁定版本与实际版本

以下 SHA 均为完整 SHA。位置与来源由根仓库 `.gitmodules:1-26` 定义；gitlink 由 `git ls-tree HEAD starVLA third_party/` 读取。`branch` 只是维护位置，复现应使用 gitlink SHA，不能 `update --remote`。

| 子模块路径 | 根仓库 gitlink = 本地实际 HEAD | 本次检出方式 | 等级 |
| --- | --- | --- | --- |
| `starVLA` | `57d508cb180d0510ac9060358009c0f6ffaac420` | 完整源码 checkout，detached HEAD | Confirmed |
| `third_party/LIBERO` | `5a2d7a27d557089747032d7a3f599568e1796844` | sparse checkout，仅源码/文档等文本，排除仿真资产 | Confirmed |
| `third_party/RoboTwin` | `8e9e66544e2e9efc5910721b8d97d6dee806fa20` | 完整仓库 checkout，未另行下载外部资产 | Confirmed |
| `third_party/openvla-oft` | `d345e2734a1b56918fb68f3d3e5f470cc45059d8` | sparse 文本 checkout | Confirmed |
| `third_party/FastWAM` | `d1e88dbcc42223cc7a07842b09d10afda644c023` | sparse 文本 checkout | Confirmed |
| `third_party/Isaac-GR00T` | `9c7e746b2cd37a810070a98ef41d290a07e806c2` | sparse 文本 checkout；上游 `NVIDIA/Isaac-GR00T` | Confirmed |

其余五个 project-maintained 子模块 URL 都是 `ShuhongLL/RawVLA.git`，分别使用 `submodules/<name>` 分支。不是从 StarVLA/LIBERO/RoboTwin 的独立上游默认分支取最新代码。证据：`.gitmodules`、`SUBMODULES.md:1-22`。**Confirmed**。

`Isaac-GR00T` 的四个嵌套子模块没有初始化，不能把 gitlink 说成实际 checkout；它们不属于本阶段主线 Qwen3-OFT/π0/π0.5 的必要运行路径：

| 嵌套路径（相对 `third_party/Isaac-GR00T/`） | 声明的 gitlink | 实际 HEAD |
| --- | --- | --- |
| `external_dependencies/LIBERO` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | Unknown：未初始化 |
| `external_dependencies/SimplerEnv` | `8a2d286c926c1371927caa7651a412b4cc331756` | Unknown：未初始化 |
| `external_dependencies/robocasa` | `d89d481ce9c76da7f179466981676e268aa842e5` | Unknown：未初始化 |
| `external_dependencies/robocasa-gr1-tabletop-tasks` | `4840e671596f93ca03651524b9f72ffb1aadfeff` | Unknown：未初始化 |

证据：`git submodule status --recursive` 的四条 `-<SHA>` 输出；来源 URL 见 `third_party/Isaac-GR00T/.gitmodules:1-12`。本地 sparse 检出不是可直接启动仿真的完整安装；后续仿真阶段需在专用运行 checkout 恢复所需资产。

## 3. 可重复的命令输出

在根 checkout 执行：

```text
$ git rev-parse HEAD
ba9c56b04da2c4931b70d8844710f5928d43775b
$ git branch --show-current
main
$ git status --porcelain=v1 --untracked-files=all
<无输出>
$ git diff --stat
<无输出>
$ git submodule status
 57d508cb180d0510ac9060358009c0f6ffaac420 starVLA (57d508c)
 d1e88dbcc42223cc7a07842b09d10afda644c023 third_party/FastWAM (heads/submodules/FastWAM)
 9c7e746b2cd37a810070a98ef41d290a07e806c2 third_party/Isaac-GR00T (9c7e746)
 5a2d7a27d557089747032d7a3f599568e1796844 third_party/LIBERO (heads/submodules/LIBERO)
 8e9e66544e2e9efc5910721b8d97d6dee806fa20 third_party/RoboTwin (8e9e665)
 d345e2734a1b56918fb68f3d3e5f470cc45059d8 third_party/openvla-oft (heads/submodules/openvla-oft)
```

子模块每条开头为空格，表示与 gitlink 一致。四个 sparse 子模块各自 `core.sparseCheckout=true`；六个直接子模块各自 `status --porcelain` 均为空。**Confirmed**。

## 4. 代码与论文版本关系

| 来源 | 本次核对版本 | 能确认什么 | 等级 |
| --- | --- | --- | --- |
| [论文 HTML v1](https://arxiv.org/html/2609.37530v1) | `arXiv:2609.37530v1`，页面标注 2026-09-29 | 标题、作者、§3.4、附录 B/D 是本次比对依据 | Confirmed |
| [官方项目主页](https://shuhongll.github.io/rawvla/) | 2026-11-16 UTC 读取的网页；无可验证版本 SHA | 关联同一论文、代码和 HF 数据集 | Confirmed：链接关联；网页版本冻结 Unknown |
| 根仓库 README | 固定 root SHA，`README.md:24-28,237-248` | 引用论文编号 `2609.37530`，没有指定论文实验对应代码 commit | Confirmed |
| 根仓库版本历史 | 当前 commit 晚于论文 v1；远端无 tag 输出 | 可以确认时间先后，不能证明当前默认配置就是论文训练配置 | Partially Confirmed |
| 论文实验所用精确代码/配置/模型产物 SHA | 未发现论文将实验绑定到本次 root/submodule SHA 的声明，也未取得训练日志 | 无法建立逐实验精确映射 | Unknown |

本次下载的论文 HTML（非数据集或权重）SHA256：`c2291e754d0e8e9516a4f48c71925c7ee9cb65f7a7bf96c95534c1bd464b41d2`。URL 使用显式 `v1`。网页处理后的文本不是作者发布源，不以其临时本地行号代替论文章节/公式标识。

当前代码不能直接认定为“论文 v1 完全一致版”：公开 LIBERO uint8 存储、Loss 默认值、RoboTwin 配对监督与正式 manifest 执行路径都需分别核对。详见四项专题报告。这里描述的是已证实的差异及待验证点，不据此推断论文结果真伪或给代码直接定性为 Bug。

## 5. 本机状态及已完成验证

本机只作审计，非作者实验环境。命令 `python -B` / `importlib.util.find_spec` / `free -h` 输出：

```text
Python 3.12.14
Linux x86_64
os.cpu_count() = 5
RAM: 17 GiB total, 16 GiB available（当次读取）
numpy=True, yaml=True
torch=False, omegaconf=False, accelerate=False, transformers=False
command -v nvidia-smi: 无输出
```

**Confirmed**：当前解释器缺少上述训练依赖。**Unknown**：宿主是否另有未暴露的 GPU；本次没有可见 NVIDIA 工具链，不能声称做过 GPU 验证。

已执行源码 AST 解析（不 import、不生成 `.pyc`），共 1,494 个 `.py`：1,493 个可解析，唯一失败是文档模板 `starVLA/docs/agent_skills/integrate-starvla-dataset/assets/templates/model2bench_interface.py:44-45` 中 `<<TODO_STATE_DIM>>` / `<<TODO_ACTION_DIM>>` 占位符。另两处 vendored RoboTwin 脚本有转义 SyntaxWarning。**Confirmed**：这不是运行测试；不能由此证明依赖、模型前向、梯度或仿真能运行，也不把模板占位符视为主训练入口语法故障。

未执行：NPZ 实物加载、RAW frontend 前向、VLA batch 前向、反向传播、优化器步进、仿真 episode、任何成功率统计。

## 6. 协作和证据口径

本环境实际启动了三个独立 Subagents：

- Agent A `dataset_audit`：Audit A → `DATA_FORMAT_AUDIT.md`。
- Agent B `loss_audit`：Audit B → `LOSS_CONFIG_AUDIT.md`。
- Agent C `eval_env_audit`：Audit C/D → `EVALUATION_PROTOCOL_AUDIT.md`、`ENVIRONMENT_AUDIT.md`。
- 主 Agent：冻结版本、交叉核对、整合与 Audit E → 本文及 `MINIMAL_REPRO_PLAN.md`。

证据等级统一：**Confirmed** = 可从固定源码、明确论文原文或本次命令/API 输出直接核验；**Partially Confirmed** = 部分链路或文档声明已核对，但未由实物/运行闭环；**Unknown** = 缺少所需文件、日志、权限或运行条件。所有资源估计都是规划值，不伪装为实测。

六份报告中的源码路径均相对本根 checkout；`starVLA/...` 前缀表示锁定的 starVLA 子模块。永久代码链接格式：`https://github.com/ShuhongLL/RawVLA/blob/<对应根或子模块SHA>/<该仓库内路径>#L<行号>`。子模块链接必须使用其自身 SHA，不使用 root SHA。
