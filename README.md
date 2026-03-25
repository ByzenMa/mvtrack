# mvtrack

`mvtrack` 是一个跨视角目标定位与匹配项目，包含两个核心子模块：

- `segmentation/`：基于文本提示（text prompt）在视频帧中分割目标，并导出每个视角下的目标区域。
- `match/`：对不同视角中的候选目标进行匹配，确保三路视角中定位到的是同一实例。

## 整体流程

1. **Segmentation 阶段**
   - 输入：同一场景同一 clip 的多视角视频流 + 文本描述。
   - 输出：每个视角中与文本相关目标的掩码/框（bbox 可由 mask 提取）。
2. **Match 阶段**
   - 输入：三个视角中已定位的目标区域。
   - 输出：跨视角实例匹配结果（确认是否为同一目标）。

## 目录结构

```text
mvtrack/
├── segmentation/      # 文本引导的视频目标分割（SAMWISE）
├── match/             # 跨视角目标匹配与 ReID 训练/推理代码
└── README.md
```

## 训练与推理说明

### 1) Segmentation（SAMWISE）

常用入口：

- 训练：`segmentation/main.py`
- 推理：`segmentation/inference_demo.py`、`segmentation/inference_mevis.py`、`segmentation/inference_ytvos.py`、`segmentation/inference_crtrack.py`

示例（在 `segmentation/` 下）：

```bash
python main.py --dataset_file crtrack --name_exp crtrack_samwise
```

> 当前仓库设置为：在 CRTrack 训练时，三个视角会在同一个 iteration 内共同参与训练，并对各视角 loss 进行求和。

### 2) Match

`match/` 子目录包含独立的训练配置与脚本（如 `match/train/run.sh`），用于跨视角重识别/匹配任务。

## 环境准备

建议使用 Python 3.10+，并根据两个子模块分别安装依赖：

- `segmentation/requirements.txt`
- `match/` 下训练脚本依赖（按实际环境补齐）

## 备注

- `segmentation` 与 `match` 可独立调试，也可串联成完整 pipeline。
- 若需复现实验，请优先核对数据路径参数（如 `--crtrack_path`、`--coco_path` 等）。
