# Audit E — 从零开始的最小复现路线

日期：2026-11-16（Asia/Shanghai）。整合三个独立 Agent 的静态证据。本阶段仅执行代码阅读、Git/JSON/YAML/API 元数据与 AST 检查；**下列安装、样本加载、模型前向、训练和仿真命令均为下一阶段方案，未执行**。

## 1. 当前可开始到哪一步

**Confirmed**：可以开始静态阅读、manifest 结构核对；拿到一个真实 NPZ 后可开始 CPU 格式检查。**Partially Confirmed**：源码具备 RAW frontend、VLA 可微图像接口、冻结主干、Loss 合并和训练入口，但尚未完成任何运行闭环。**Unknown**：当前依赖组合下真实 batch、梯度、单次仿真能否通过及最低显存。

当前审计解释器有 NumPy/PyYAML，无 Torch、OmegaConf、Accelerate、Transformers，也没有可见 `nvidia-smi`。训练入口强制 DeepSpeedPlugin，而 `environment-libero.yml` 没有列 DeepSpeed，入口还import未在该YAML声明的wandb；因此当前工作区不能直接启动最小 GPU 训练。证据：`REPO_SNAPSHOT.md` §5；`train_starvla.py:34,51-52`、`environment-libero.yml:14-63`、`starVLA/requirements.txt:23,27`。**Confirmed**。

**最推荐先做实验**：单个 LIBERO RAW NPZ 的 CPU schema/dtype/range 检查，再检查该文件对应的六 endpoint × 六 burst 索引。无需 GPU、仿真资产或 VLA 权重；若调用正式 Dataset 做动作归一化，还需要对应 backbone 的真实 statistics JSON。不要先下载整个数据集。

版本固定为 root `ba9c56b04da2c4931b70d8844710f5928d43775b`、starVLA `57d508cb180d0510ac9060358009c0f6ffaac420`。以下命令统一约定：

```bash
export RAWVLA_ROOT=/absolute/path/to/RawVLA
export AUDIT_ROOT=/absolute/path/to/research_audit
export NPZ_FILE=/absolute/path/to/one/libero_raw_episode.npz
export SINGLE_NPZ_DIR=/absolute/path/to/a_directory_containing_only_that_npz
export PYTHONPATH="$RAWVLA_ROOT/starVLA:$RAWVLA_ROOT"
export PYTHONDONTWRITEBYTECODE=1
```

以上路径需要替换，不能把占位值当成已存在的文件。本次 checkout 的四个可选/仿真子模块采用 sparse 文本检出；后续安装或仿真应使用独立运行 checkout，恢复所需 assets。所有运行日志、临时配置、probe 脚本和结果放 `AUDIT_ROOT`；不在本阶段修改官方源文件。资源数值均为规划估计，非实测门槛。

## 2. 九步路线与验收

| 步骤 | 目的 | 依赖 | 预计资源/耗时 | 验收标准 | 失败后先查 |
| --- | --- | --- | --- | --- | --- |
| 1 静态阅读 | 固定代码、画清接口与配置继承 | Git、Python 标准库，PyYAML 可选 | CPU；RAM <1GB；约1-2小时阅读 | SHA一致、diff为空、能解释 NPZ→VLA 和 Loss 系数 | 是否用错上游/子模块分支、未初始化、不同 SHA |
| 2 无 GPU 格式预检 | 不解压整轨迹，检查 ZIP/NPY头与大小 | 一个已有 NPZ、Python 标准库 | CPU；通常 <100MB；秒级 | 字段、dtype声明、HWC/N和解压大小可记录；无 object dtype | 下载不完整、非NPZ、字段版本不同、episode过大 |
| 3 单 NPZ 加载 | 核实实际 dtype/range/metadata 与索引 | NumPy、可容纳解压数据的内存 | LIBERO建议空闲2-4GB；秒至分钟 | uint8 RAW、N一致、RGB顺序合同、可复算6×6索引 | `allow_pickle=False`、N不匹配、uint8被误当RAW10、metadata缺失 |
| 4 单帧 RAW frontend | 验证无权重 CPU RAW→RGB | CPU Torch、NumPy、官方 baselines 源码 | 2-4 CPU核，RAM约2-4GB；秒至分钟 | 输出`[1,3,H,W]`、finite、[0,1]；测试确实执行 | Torch缺失被skip、HWC/CHW混淆、FFT输入dtype/尺寸 |
| 5 一个 batch 前向 | 真实 Dataset→frontend→frozen Qwen3-OFT | StarVLA环境、一个NPZ、stats、policy+base权重 | 单GPU优先A10080GB；主存32-64GB，加载检查建议64GB；启动数分钟 | list[dict]、每view `[6,6,H,W,3]`、仅frontend可训练、loss有限 | policy/base/stats不匹配、离线文件缺失、显存、图像fallback |
| 6 梯度链 | 确认action loss真的到达ISP | 第5步；保留autograd | 与5相同或更高；一次反传 | action-only梯度finite且非零、主干grad为空；递归历史有梯度 | PIL/NumPy detach、no_grad、hard clipping、误冻结、断开BPTT |
| 7 最小训练 | 验证optimizer/launcher/save链 | 第6步、DeepSpeed+CUDA工具链、匹配Accelerate | 1 GPU；先1 optimizer step，后10步；数分钟至数十分钟 | runtime accumulation明确；只更新frontend；保存可重载产物 | DS缺失/配置、warmup、累积数不生效、checkpoint格式 |
| 8 单任务单episode | 验证policy server→client→simulator | 两套环境、RoboTwin assets、模型/附属配置；严格单seed另需一份指令JSON | GPU+Vulkan；RAM32-64GB；资产约30GB；首次编译更久 | 恰好一个policy episode；完整seed/成功失败日志；模型动作进入环境 | expert重试、指令episode与seed混淆、EGL/Vulkan、端口、CuRobo |
| 9 完整Baseline | 验证公平预算与论文正式评估矩阵 | 前8步通过、作者配置/产物映射、manifest adapter、各ISP产物 | 多天至更久；按单episode计时估算；训练参考A10080GB | 逐case覆盖完整、同seed同光照、同2k训练预算、报告分域成功率 | 跳case、跑错easy/hard矩阵、缺权重、8k默认文件冒充2k |

表中接口与字段由 `DATA_FORMAT_AUDIT.md` A4-A7 确认；训练合同由 `LOSS_CONFIG_AUDIT.md` §2-4 确认；单episode参数由 `EVALUATION_PROTOCOL_AUDIT.md` C2-C4 确认。运行资源与结果属于 **Unknown，待实测**，不能因命令可解析就宣布验收通过。

## 3. 步骤1：静态阅读与冻结版本

目的：先知道自己要复现哪份代码。无 GPU、数据、权重要求。

```bash
git -C "$RAWVLA_ROOT" rev-parse HEAD
git -C "$RAWVLA_ROOT" status --porcelain=v1 --untracked-files=all
git -C "$RAWVLA_ROOT" submodule status --recursive
rg -n 'extends:|exposure_loss_weight|chroma_prior_weight|chroma_supervision_weight|action_loss_weight' "$RAWVLA_ROOT/configs"
rg -n 'raw_burst_float32|observation_indices|burst_indices|def collate_fn' "$RAWVLA_ROOT/starVLA/starVLA/dataloader/rawvla_npz_datasets.py"
rg -n 'total_loss =|exposure_weight|chroma_weight|chroma_supervision_weight|Accelerator\(' "$RAWVLA_ROOT/starVLA/starVLA/training/train_starvla.py"
```

阅读顺序：`configs/libero/qwen3_oft.yaml`→两个base→`rawvla_npz_datasets.py`→`qwen_tensor_vision.py`→`registry.py`→`baselines/rawvla/rawvla.py`→`QwenOFT.py`→trainer。**Confirmed**：对应函数和行号详见数据/Loss报告。验收时能解释两种“6”与动作 horizon 不同，也能区分论文 .001 与当前 .01。若失败，先核固定版本，不通过猜测补全接口。

从空机器开始，下一阶段可在独立目录克隆，并只初始化所需模块；不要用`update --remote`漂移版本：

```bash
git clone --filter=blob:none --no-recurse-submodules https://github.com/ShuhongLL/RawVLA.git "$RAWVLA_ROOT"
git -C "$RAWVLA_ROOT" checkout --detach ba9c56b04da2c4931b70d8844710f5928d43775b
git -C "$RAWVLA_ROOT" submodule update --init starVLA third_party/LIBERO third_party/openvla-oft
# 到RoboTwin仿真阶段才额外初始化third_party/RoboTwin；不需要先初始化全部可选栈。
```

## 4. 步骤2：只读 NPZ 头，不加载数组

目的：先确认实际样本格式并估算完整解压内存。依赖仅 Python 标准库和一份下一阶段获准准备的 NPZ。**本阶段没有下载样本**。

```bash
python -B - <<'PY'
import ast, os, struct, zipfile
p=os.environ['NPZ_FILE']
with zipfile.ZipFile(p) as z:
    total=0
    for info in z.infolist():
        assert info.filename.endswith('.npy'), info.filename
        with z.open(info) as f:
            assert f.read(6)==b'\x93NUMPY'
            version=tuple(f.read(2))
            assert version in {(1,0),(2,0),(3,0)}, version
            n=2 if version==(1,0) else 4
            length=struct.unpack('<H' if n==2 else '<I',f.read(n))[0]
            assert length < 1024*1024, 'abnormally large NPY header'
            h=ast.literal_eval(f.read(length).decode('utf-8' if version==(3,0) else 'latin1'))
        print(info.filename,h,'uncompressed_bytes=',info.file_size)
        total+=info.file_size
    print('total_uncompressed_bytes=',total)
    assert total <= 1024**3, 'choose a smaller episode or explicitly budget more RAM before step 3'
PY
```

这只确认文件头，实际数值范围要到步骤3检查。LIBERO期待 `agentview_raw_uint8/wrist_raw_uint8`；RoboTwin期待三路 `*_raw_float32`，不能要求原始 NPZ 存在 `raw_burst_float32`。证据：Dataset `:244-265,281-294`。**Confirmed（源码合同）；实际文件待验**。

验收：保存字段/shape/dtype声明及压缩、解压大小；若超过1GiB先选较小episode或提高明确内存预算。不能用 `mmap_mode='r'` 假设压缩NPZ变成逐帧内存映射。证据：Dataset `_load_episode:130-135` 一次读所有字段。

## 5. 步骤3：加载单 NPZ、检查真实值与采样

```bash
python -B - <<'PY'
import hashlib, json, os, zipfile
from pathlib import Path
import numpy as np
p=Path(os.environ['NPZ_FILE'])
with zipfile.ZipFile(p) as f:
    assert sum(i.file_size for i in f.infolist())<=1024**3
digest=hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
print('sha256=',digest.hexdigest())
keys=('agentview_raw_uint8','wrist_raw_uint8')
with np.load(p,allow_pickle=False) as z:
    print('files=',z.files)
    meta=json.loads(str(z['metadata_json'].item())); print('metadata=',meta)
    n=z['actions'].shape[0]; assert n>0
    assert z['robot_states'].shape[0]==n
    for k in keys:
        a=z[k]
        assert a.dtype==np.uint8 and a.shape==(n,256,256,3),(k,a.dtype,a.shape)
        print(k,a.dtype,a.shape,int(a.min()),int(a.max()))
        del a
    t=int(np.random.default_rng(42).integers(n))  # 对应官方index=0
    ep=np.clip(t-8*np.arange(5,-1,-1),0,n-1)
    burst=np.clip(ep[:,None]-np.arange(5,-1,-1)[None,:],0,n-1)
    act=np.clip(np.arange(t,t+8),0,n-1)
    print('index0 current=',t,'endpoints=',ep,'burst=',burst,'actions=',act)
    assert burst.shape==(6,6) and burst[-1,-1]==t
PY
```

此脚本兼容正式环境Python3.10，但因无样本未执行。RoboTwin变体需按三camera真实key检查float32、finite、min≥0/max≤1，**不得再除255**；stride=5、action_offset=1，见数据报告A4/A6。

若有配对RGB文件，再核对相对路径、actions/states及帧数一致。颜色顺序不能只由shape证明；还应对照writer和小型预览人工确认。正式Dataset测试必须提供真实stats，不用伪造min/max证明动作归一化正确。**Partially Confirmed**：预期合同已知，所有实物验收仍待运行。

## 6. 步骤4：单帧 RAW frontend，无 VLA 权重

下一阶段在单独CPU环境准备Torch；不必为此先装完整simulator。例如使用系统兼容的Python创建venv，再安装CPU wheels（命令仅为下一阶段计划）：

```bash
python3 -m venv "$AUDIT_ROOT/env-cpu"
source "$AUDIT_ROOT/env-cpu/bin/activate"
python -m pip install 'numpy==1.26.4' pillow omegaconf
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
```

以下只使用已存在的单个NPZ：

```bash
cd "$RAWVLA_ROOT"
python -B - <<'PY'
import os, numpy as np, torch
from baselines.rawvla import RAWVLA
with np.load(os.environ['NPZ_FILE'],allow_pickle=False) as z:
    a=z['agentview_raw_uint8'][0].astype(np.float32)/255.0
x=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).unsqueeze(0).contiguous()
assert x.ndim==5 and x.shape[:3]==(1,1,3)
torch.manual_seed(42)
m=RAWVLA(luma_state_dim=64).eval()
with torch.no_grad(): y=m(x)
assert y.rgb.shape==(1,3,a.shape[0],a.shape[1])
assert torch.isfinite(y.rgb).all() and y.rgb.min()>=0 and y.rgb.max()<=1
print('single-frame PASS',y.rgb.shape,y.state.shape)
PY
python -B -c 'import torch; print(torch.__version__)'
python -B -m baselines.rawvla.smoke_test
```

证据：`RAWVLA.__init__:581-609`、`forward:1099-1128`；`smoke_test.py:11-76,79-95,112-183`。**Confirmed（接口）**。单帧K=1会触发denoiser单帧fallback，不能证明六帧降噪有效；现有smoke另覆盖多帧与梯度，但不覆盖VLA action链。缺Torch时smoke可skip并返回0，因此验收必须看实际执行、assertions，不能只看退出码。

## 7. 步骤5-6：一个真实 batch 前向与 action-only 梯度

选择 **LIBERO Qwen3-OFT** 作为首个完整链路，原因是配置与公共checkpoint/base获取路径清晰，不要求先完成PI格式转换或RoboTwin仿真。依赖：`environment-libero.yml`主环境及本地Qwen3 base、OFT policy、stats；详情见 `ENVIRONMENT_AUDIT.md`。`.pt`约9.785GB、base仓库约8.887GB，至少预留约25GB权重存储及额外环境空间。推理/反传最低显存未知；优先用论文同款A10080GB，不能因VLA被冻结就忽略图像输入梯度所需激活内存。

在下一阶段GPU机器上切换到完整环境，退出CPU venv，按官方安装路径准备；这些安装命令未在本阶段执行：

```bash
cd "$RAWVLA_ROOT"
conda env create -f environment-libero.yml
conda activate starvla-libero
python -m pip install --no-deps --no-build-isolation -e ./third_party/LIBERO
python -m pip install --no-deps -e ./third_party/openvla-oft
python -m pip install --no-deps -e ./starVLA
export LIBERO_CONFIG_PATH="$AUDIT_ROOT/local/libero"
python scripts/setup/configure_libero.py --datasets "$AUDIT_ROOT/local/libero-datasets"
# 训练入口另需deepspeed/wandb，见步骤7；完成依赖后保存实际pip/conda环境快照。
```

证据：`ENVIRONMENTS.md:25-43`与`configure_libero.py`的`LIBERO_CONFIG_PATH`处理。运行前先确认单文件目录确实只含一份RAW NPZ，避免误遍历整个训练集。把下面probe在下一阶段保存为 `$AUDIT_ROOT/batch_probe.py`，不改官方文件。其作用是验证真实权重覆盖、冻结、Dataset接口和一批原生action loss，不执行optimizer.step：

```python
import importlib.util, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from starVLA.config_loader import load_config
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

root=Path(os.environ['RAWVLA_ROOT'])
torch.manual_seed(42)
cfg=apply_config_compat(load_config(root/'configs/libero/qwen3_oft.yaml'))
cfg.datasets.vla_data.cache_root=os.environ['SINGLE_NPZ_DIR']
assert len(list(Path(cfg.datasets.vla_data.cache_root).rglob('*.npz')))==1
spec=importlib.util.spec_from_file_location('audit_npz',root/'starVLA/starVLA/dataloader/rawvla_npz_datasets.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
ds=module.RAWVLALightNPZDataset(cfg.datasets.vla_data)
batch=next(iter(DataLoader(ds,batch_size=1,num_workers=0,collate_fn=module.collate_fn)))
assert isinstance(batch,list) and len(batch)==1
assert len(batch[0]['raw_burst_float32'])==2
for a in batch[0]['raw_burst_float32']:
    assert a.shape[:2]==(6,6) and a.shape[-1]==3 and a.dtype==np.float32
    assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
model=build_framework(cfg)
# 与官方global state-dict加载语义相同，并额外核实所有frozen参数被checkpoint覆盖。
state=torch.load(str(cfg.trainer.pretrained_checkpoint),map_location='cpu',weights_only=True)
assert isinstance(state,dict) and all(isinstance(k,str) for k in state)
model=TrainerUtils.freeze_backbones(model,str(cfg.trainer.freeze_modules))
frozen={n for n,p in model.named_parameters() if not p.requires_grad}
assert frozen and frozen<=set(state), sorted(frozen-set(state))[:20]
incompat=model.load_state_dict(state,strict=False)
assert not (set(incompat.missing_keys)&frozen)
assert not incompat.unexpected_keys, incompat.unexpected_keys[:20]
del state
assert all(n.startswith('raw_frontend.') for n,p in model.named_parameters() if p.requires_grad)
model=model.cuda().eval()  # frozen VLA需eval；eval本身不关闭autograd
model.raw_frontend.train()
with torch.autocast('cuda',dtype=torch.bfloat16):
    out=model(batch)
print({k:float(v.detach()) for k,v in out.items() if torch.is_tensor(v) and v.numel()==1})
action=out['action_loss']; assert action.ndim==0 and torch.isfinite(action)
assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))
if os.environ.get('PROBE_BACKWARD','0')=='1':
    assert action.requires_grad
    action.backward()  # 只用动作loss，避免auxiliary独自有梯度造成假通过
    grads=[p.grad for p in model.raw_frontend.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    norm=torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
    assert norm>0
    assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
    print('action_to_ISP_gradient_norm=',float(norm))
print('batch probe PASS')
```

执行命令：

```bash
cd "$RAWVLA_ROOT/starVLA"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PROBE_BACKWARD=0 python -B "$AUDIT_ROOT/batch_probe.py"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PROBE_BACKWARD=1 python -B "$AUDIT_ROOT/batch_probe.py"
```

**Partially Confirmed**：probe按已读接口编写，尚未执行，真实checkpoint是否是平铺state dict未检验。覆盖断言失败时检查产物格式/模型revision/配置，不直接删除断言。官方 `TrainerUtils.load_pretrained_backbones:253-305` 的global load为 `strict=False`，仅出现“loaded”不足以证明匹配。此probe不import训练入口，避免仅为前向测试就构造强制DeepSpeedPlugin。

第6步还要验证BPTT历史而非只看最终参数有grad。下面是在无VLA权重的frontend上，用小幅非饱和合成输入保留各endpoint state梯度的独立CPU检查；它与上方真实action-only梯度应分别验收：

```bash
python -B - <<'PY'
import os, torch
from pathlib import Path
from starVLA.config_loader import load_config
from starVLA.model.modules.raw_frontend import build_raw_frontend
torch.manual_seed(42)
c=load_config(Path(os.environ['RAWVLA_ROOT'])/'configs/libero/qwen3_oft.yaml')
f=build_raw_frontend(c.framework.raw_frontend).train()
traces=[]
def keep_state(module, inputs, output):
    output.state.retain_grad()
    traces.append(output)
hook=f.model.register_forward_hook(keep_state)
x=(0.05+0.10*torch.rand(1,6,6,3,64,64)).requires_grad_()
y=f(x); (y*torch.randn_like(y)).mean().backward(); hook.remove()
assert len(traces)==6
for i,o in enumerate(traces[:-1]):
    g=o.state.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0, i
assert x.grad is not None and torch.isfinite(x.grad).all()
print('ISP recurrent-state gradients PASS')
PY
```

现有源码`registry.py:332-337`没有detach只证明静态链路，不能代替数值检查。完整VLA probe如需进一步对早期endpoint的action-only梯度定位，可用相同hook，仅改 `AUDIT_ROOT` probe。**Unknown：参数action梯度与6步历史梯度本次均未执行**。

## 8. 步骤7：最小1-step训练，再考虑10-step

先在下一阶段验证DeepSpeed：`python -B -c 'import deepspeed,torch,accelerate,wandb; print(deepspeed.__version__,torch.__version__,accelerate.__version__)'`。基准版本线索是 `starVLA/requirements.txt:23,27` 的 `deepspeed==0.16.9`与`wandb`；在独立训练环境以 `python -m pip install 'deepspeed==0.16.9' wandb` 为候选安装，再检查编译/版本兼容性，保存实际解算结果后运行。不修改官方YAML掩盖依赖缺口。`ACCELERATE_USE_DEEPSPEED=false`不能绕过代码显式传入plugin。

```bash
cd "$RAWVLA_ROOT/starVLA"
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1 CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes 1 starVLA/training/train_starvla.py \
  --config_yaml ../configs/libero/qwen3_oft.yaml \
  --datasets.vla_data.cache_root "$SINGLE_NPZ_DIR" \
  --datasets.vla_data.num_workers 0 \
  --trainer.max_train_steps 1 --trainer.num_warmup_steps 0 \
  --trainer.logging_frequency 1 --trainer.save_interval 1000000 \
  --run_root_dir "$AUDIT_ROOT/runs" --run_id published_defaults_one_step
```

**Confirmed（静态语义）**：这是当前发布默认Loss的一步训练，曝光=.01，不是论文文字=.001。入口读取位置见 `train_starvla.py:985-995,640-661`；总梯度与component梯度输出见`:669-708`；final frontend保存见`:914-937`。虽然save_interval很大，最终仍会写checkpoint。

验收：runtime accumulation=1、`config.full.yaml`可读；action-to-ISP梯度非零、loss有限；只更新frontend；final产物能以registry的严格接口重载并在同一probe重复前向。第一次只1步；通过后将max steps改10观察数值稳定，不以10步loss下降宣称复现性能。若DeepSpeed JSON覆盖环境变量，需以日志为准；YAML accumulation=8本身不保证runtime=8。

要比较论文文字，另开独立 `run_id=paper_text_one_step` 并加 `--framework.raw_frontend.exposure_loss_weight 0.001`；其它backbone覆盖清单见Loss报告§7。不要将改变后的配方继续标成“发布默认”。作者真实论文配置仍 **Unknown**。

## 9. 步骤8：自定义单任务、单policy episode

目的：先验证机器人仿真闭环，允许episode成功或失败，重点是接口通。此处用Qwen3 RoboTwin RGB policy；它与步骤5的LIBERO policy是不同权重，不可混用。两套Python环境、资产、checkpoint附属config/stats、Qwen3base缺一不可。

严格命令与解释见 `EVALUATION_PROTOCOL_AUDIT.md` C3；核心终端如下，所有路径需换成真实已准备文件：

```bash
# 终端A：starvla-libero环境
cd "$RAWVLA_ROOT/starVLA"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -B deployment/model_server/server_policy.py \
  --ckpt_path "$QWEN3_ROBOTWIN_CKPT" --port 5694 --use_bf16 --idle_timeout -1 \
  --config_override "framework.qwenvl.base_vlm=$QWEN3_BASE"
```

```bash
# 终端B：先export上方路径变量，等待A ready后运行；ROBOTWIN_PYTHON指向独立simulator解释器
cd "$RAWVLA_ROOT"
ROBOTWIN_PATH="$RAWVLA_ROOT/third_party/RoboTwin" \
ROBOTWIN_PYTHON=/absolute/path/to/robotwin/bin/python \
ROBOTWIN_TEST_NUM=1 ROBOTWIN_EVAL_SEEDS=0 ROBOTWIN_IMAGE_OBS_KEY=render_rgb \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
bash starVLA/examples/simBenchmarks/Robotwin/eval_files/eval.sh \
  grab_roller demo_clean qwen3_rgb_smoke 0 0 "$QWEN3_ROBOTWIN_CKPT" 5694 127.0.0.1
```

`QWEN3_ROBOTWIN_CKPT` 和 `QWEN3_BASE` 需由用户下一阶段明确设置。默认仍可能额外执行expert/planner检查；若要求**总共仅一个policy rollout且无expert rollout**，按C3额外提供 `instructions/episode0.json` 并export `ROBOTWIN_USE_DATASET_INSTRUCTIONS=1`、`ROBOTWIN_POLICY_AUDIT_DATA_ROOT`、`START_EPISODE=0`。seed列表与episode索引各有用途，不能互换。

simulator环境准备命令同样只留待下一阶段执行：`conda env create -f environment-robotwin2.yml`，激活robotwin，先链接已有assets（`ROBOTWIN_ASSETS_SOURCE=/existing/assets bash scripts/setup/setup_robotwin_assets.sh`），再执行`ROBOTWIN_PYTHON=python bash third_party/RoboTwin/script/_install.sh`。没有已有assets时先单独规划约30GB资产获取，不能将这条安装混入当前只读审计。证据：`ENVIRONMENTS.md:53-72`。

验收日志包括实际seed=0、一次policy Success/Fail、14D动作及结果文件；退出后关闭终端A服务器。若expert异常不断重试，不持续等待：先检查固定指令分支与资产/任务可行性。**Partially Confirmed**：接口已查、命令未跑。默认可能生成结果/视频，后续应在专用运行checkout执行；本阶段未运行这些会写文件的脚本。

README的`one adjust_bottle`默认实际为200 rollouts，且不读取3250manifest；这里的单episode不等于那条示例，也不等于论文协议。**Confirmed**：simulator的`rgb`实际是uint8 RAW可视化；真正renderer RGB是`render_rgb`，未量化RAW是`raw_linear`（`camera.py:384-391`）。RAW frontend仿真应在RGB闭环通过后再用`ROBOTWIN_IMAGE_OBS_KEY=raw_linear`独立验收float32 transport、状态重置和前端权重，见C4。

## 10. 步骤9：完整Baseline复现，分成可审核的四层

**第一层：固定Default ISP。** 无需训练ISP，先在一款frozen policy上验证LIBERO完整manifest。下一阶段、模型/资产已就绪后，以下命令选择一款policy、单GPU，避免脚本默认六模型/八GPU矩阵：

```bash
cd "$RAWVLA_ROOT"
PY=python GPU_IDS=0 JOBS_PER_GPU=1 \
LIBERO_MODELS=Qwen3-VL-OFT-LIBERO-4in1 \
LIBERO_SUITES='libero_spatial libero_object libero_goal libero_10' \
NUM_TRIALS=50 RAW_FRONTEND=none RAWVLA_BENCH_REPRESENTATION=default_isp \
RAWVLA_BENCH_MANIFEST="$RAWVLA_ROOT/experiments/simulation/libero/manifests/libero_manifest_50_init_states_10000_rollouts.json" \
LOG_ROOT="$AUDIT_ROOT/eval/default_isp_qwen3" \
bash experiments/simulation/libero/scripts/run_libero_bench.sh
```

证据：wrapper`:19-37`、model runner`:33-66,135-145,293-317`。**Confirmed（参数与静态路径）；Unknown（运行耗时与结果）**。先另以`LIBERO_SUITES=libero_spatial MAX_TASKS=1 NUM_TRIALS=1 MAX_NEW_EPISODES_PER_RUN=1`限量校验，仍须审阅client的manifest过滤与输出，不能仅凭变量名认定恰好一个manifest entry。

**第二层：RawVLA一个backbone。** 用已选定且明确标为published_defaults或paper_text的配方训练2000 optimizer steps，batch=1、显式runtime accumulation=8，保留真实config/CLI/seed/环境锁与梯度日志。训练命令沿用步骤7，去掉smoke的1step/warmup0覆盖，设正式run_id；baseline公平性另需对齐论文D.1/D.2（各ISP共2000步）。再用以下命令评估训练产物：

```bash
cd "$RAWVLA_ROOT"
PY=python GPU_IDS=0 JOBS_PER_GPU=1 \
LIBERO_MODELS=Qwen3-VL-OFT-LIBERO-4in1 \
NUM_TRIALS=50 RAW_FRONTEND=rawvla RAWVLA_BENCH_REPRESENTATION=raw \
RAW_FRONTEND_CHECKPOINT="$AUDIT_ROOT/runs/FORMAL_RUN/final_model/pytorch_model.pt" \
LOG_ROOT="$AUDIT_ROOT/eval/rawvla_qwen3" \
bash experiments/simulation/libero/scripts/run_libero_bench.sh
```

`FORMAL_RUN`是待准备的真实2000-step产物目录，不能拿one-step产物填入正式比较。证据：frontend checkpoint读取`registry.py:610-644`、runner`:162-176,191-228`、trainer最终保存`:914-937`。**Partially Confirmed**：格式接口静态匹配，但尚未实际重载。

**第三层：全部ISP与主backbones。** 论文LIBERO主线3款policy×(Default ISP+4个comparison ISP+RawVLA)×10000=180000 policy rollouts；RoboTwin主线2款×6方法×3250=39000。计数是按当前论文主线范围的算术规划，未含复跑/ablation。证据：root README`:151-176`、论文§5/D.2和C报告。单episode实测若需T秒，时间约为总数×T/实际并发数，再加训练/启动开销；不能凭脚本默认GPU列表推断作者实际用了8卡。

**Confirmed**：当前LIBERO脚本对DarkISP/RAM/RAWild默认寻找8k命名文件，RAW-Adapter为4k或8k（`run_libero_models.sh:178-187`）；论文D.1/D.2是统一2k预算。文件名不证明实际训练预算，也不证明产物存在。`RAW_FRONTEND_CHECKPOINT`的任意路径覆盖目前仅RawVLA分支使用（`:166-170`）。正式comparison需要下一阶段在独立launcher显式指定经过核实的2000-step产物，或审查作者提供产物；不要改名8k冒充2k。主配置没有直接提供所有comparison ISP训练命令，**Unknown：当前发布入口能否原样完成公平训练矩阵**，需Reproduction/Training Agent补足独立配置并审查。

**第四层：RoboTwin manifest驱动正式协议。** 当前没有已验证的一键命令。不能把`full`或easy/hard矩阵当作正式protocol。先运行C5的纯JSON覆盖检查，再由Evaluation Agent在 `AUDIT_ROOT` 中实现独立manifest adapter，逐条传递实际dataset seed、指令episode、domain/sample-unit/noise/white level，核对实际camera metadata，并支持失败/错误/去重与断点续跑。接口合同见C4。**Unknown：adapter尚未实现及运行验证**，本只读阶段不修改官方runner来填补它。

完整验收需每个policy/方法3250唯一案例、每域650、无静默跳过；所有方法同seed/指令/光照/动作归一化；保存逐episode结果而非仅shell DONE/FAILED marker。没有作者最终前端checkpoint与逐实验配置对应关系时，只能报告新实验结果，不能承诺与论文逐值一致。

## 11. 下一阶段需要准备的最小物料与Agent

| 目标 | 最小物料 | 暂不需要 |
| --- | --- | --- |
| CPU schema/范围 | 一份LIBERO RAW NPZ，记录HF revision与文件SHA256 | GPU、VLA权重、完整训练集、RoboTwin assets |
| 正式Dataset样本 | 上项+选定backbone的stats JSON，Torch/Pillow/OmegaConf | 大模型权重本体、simulator |
| CPU frontend | 上项+CPU Torch；RAWVLA随机初始化 | 预训练VLA/ISP权重 |
| Qwen VLA前向/训练 | Qwen3 LIBERO OFT policy、同级stats/config、Qwen3base/processor；GPU；训练另加DeepSpeed | RoboTwin双环境、整套HF数据 |
| RoboTwin单episode | RoboTwin policy及其base/config/stats、simulator资产与Vulkan；严格无expert时一份指令JSON | 632条训练轨迹、3250预渲染测试数据（该数据本来未发布） |
| 正式全量比较 | 锁定训练集revision、完整协议adapter、每方法/每policy可验证产物及运行配置 | 未先测量就安排的大规模并发 |

资源来源和许可明细见 `ENVIRONMENT_AUDIT.md`；无license字段的权重不能由根代码MIT许可替代。报告资源数字是可用来做预算的元数据，不是下载请求。

建议Work后续分工：

1. **Dataset Validation Agent**：只准备一个批准的小样本与stats，完成步骤2-3、采样/配对核对，决定是否扩大数据量。
2. **Environment / Runtime Agent**：锁定训练与simulator两环境，验证DeepSpeed、CUDA、EGL/Vulkan、源checkout完整性；先不启动大实验。
3. **Gradient / Training Reproduction Agent**：步骤4-7，记录action-only与BPTT梯度；隔离paper_text和published_defaults，核实各ISP2k预算。
4. **Evaluation Protocol Agent**：实现并审查RoboTwin manifest adapter，先1entry→单任务250entry→3250；处理PI0/PI05动作/state/image合同。
5. **Research / Provenance Agent**：查清作者论文run config、CLI、最终ISP权重与代码/数据revision映射、缺失许可；准备具体问题和证据。本次未向作者发送消息。

先完成前三项的微型验收再安排全量实验。本阶段交付建立的是可核验路线与明确阻碍，不是训练或成功率复现完成声明。
