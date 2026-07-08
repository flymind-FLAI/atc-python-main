# ATC Transcribe

Real-time ATC (Air Traffic Control) speech transcription for macOS, powered by a fine-tuned Whisper model on Apple Silicon.

## 项目结构

```
atc-python/
├── atc_app.py              # 主程序：桌面 GUI，麦克风实时转录
├── atc_player.py           # 工具：播放 ATCO2 数据集样本到音频输出
├── realtime_atc.py         # 工具：无麦克风，终端模拟实时转录 + WER
├── eval_atc_wer.py         # 评测：ATC 模型 vs baseline WER 对比
├── compare_atc_models.py   # 评测：多个 MLX 模型 WER + 延迟横向对比
├── compare_result.txt      # v2 vs v3 模型对比结果存档
├── correction_result.txt   # 纠错实验结果存档
├── models/                 # 模型权重（git ignored）
│   ├── whisper-atc-mlx/    # MLX 格式，主力推理
│   └── whisper-atc-weights/# HuggingFace 格式，评测用
└── data/                   # 数据集（git ignored）
    └── atco2_test/         # ATCO2 1小时测试集，871样本
```

## 模型

| 路径 | 描述 |
|------|------|
| `models/whisper-atc-mlx` | whisper-large-v2 fine-tuned on ATCO2+ATCOSIM，MLX格式，主力模型，WER 15.35% |
| `models/whisper-atc-weights` | 同模型的 HuggingFace transformers 格式，供 `eval_atc_wer.py` 使用 |

训练数据以布拉格/Ruzyne空域为主，覆盖欧洲主要航司（Lufthansa、Ryanair、CSA、Belavia等）和捷克导航点。

## 数据集

`data/atco2_test/` — ATCO2 1小时测试集，871个样本，HuggingFace Arrow 格式（`load_from_disk` 加载，不需要网络）。全量 WER 15.1%。

`data/ATCO2-LIDdataset-v1_beta/` — ATCO2 语言识别数据集，约11,889条 wav 文件。包含 CZEN（布拉格）、FREN（法国）、GEEN（德国）、EN-AU（悉尼）子集。**无可靠 ground truth**（转录为 ASR confusion network 格式），不适合算 WER，可用于测试不同口音/空域下的实际转录效果。

## 脚本

### `atc_app.py` — 主程序（麦克风实时转录）

```bash
python atc_app.py
```

macOS Tkinter 桌面应用，从麦克风实时录音并转录。功能：
- 基于静音检测的分句提交（`SILENCE_THRESH = 0.005`，最长60秒）
- 说话人区分（ATC/PLT），使用 resemblyzer VoiceEncoder
- ICAO音标字母 Title Case（Alpha、Bravo…）、ATC缩写大写（ILS、QNH…）
- 数字词转阿拉伯数字并括号标注（`nine zero nine` → `nine zero nine (909)`，`two thousand five hundred` → `two thousand five hundred (2500)`）
- 支持 `decimal`/`point` 转小数点（`one two one decimal five` → `121.5`，`Mach decimal eight` → `0.8`）

**输入设备选择：** 启动后在下拉菜单选择麦克风或 BlackHole 虚拟设备。

**依赖：** `mlx-whisper`、`resemblyzer`、`sounddevice`、`tkinter`

---

### `atc_player.py` — ATCO2 数据集播放器

```bash
python atc_player.py
```

将 ATCO2 测试集样本通过系统音频输出逐条播放（样本间停顿2秒）。配合 `atc_app.py` 使用：将系统音频路由到 BlackHole，让 app 接收。

**用法：** 在 Audio MIDI Setup 里创建 Multi-Output Device（BlackHole 2ch + 扬声器），系统输出选该设备，app 输入选 BlackHole 2ch，然后运行此脚本。

---

### `realtime_atc.py` — 无麦克风实时模拟

```bash
python realtime_atc.py
```

从 ATCO2 测试集读取样本，模拟实时流式转录，在终端并排显示 REF（参考文本）和模型输出，并实时计算 WER。不需要麦克风，适合快速验证模型效果。

---

### `eval_atc_wer.py` — WER 基准评测

```bash
python eval_atc_wer.py
```

对比 baseline（whisper medium.en）和 fine-tuned ATC 模型在 ATCO2 测试集上的 WER，逐样本输出对比。使用 HuggingFace transformers 格式模型（`models/whisper-atc-weights`）和 CPU 推理，速度较慢。

配置项（脚本顶部）：
- `N_SAMPLES` — 评测样本数，`None` 为全量871条
- `BASELINE_MODEL` — baseline 模型名（默认 `medium.en`）

---

### `compare_atc_models.py` — 多模型横向对比

```bash
python compare_atc_models.py
```

在 ATCO2 测试集上对比多个 MLX 模型的 WER 和推理延迟（RTF），逐样本输出差异。`MODELS` 字典里配置模型路径和名称。

结果见 `compare_result.txt`（v2 vs v3 的对比历史）。

---

## 版本

- `v1.0-baseline` tag — 稳定基线，无 LLM 纠错，WER 15.35%

## 依赖安装

```bash
pip install mlx-whisper resemblyzer sounddevice soundfile jiwer rapidfuzz \
            transformers torch datasets
```

需要 macOS + Apple Silicon（MLX 推理）。

## 本地路径配置

默认脚本使用仓库内的相对路径：

- `models/whisper-atc-mlx`
- `models/whisper-atc-weights`
- `models/whisper-atc-v3-mlx`
- `data/atco2_test`

这些目录被 `.gitignore` 排除，需要在本机放置。也可以用环境变量覆盖模型路径：

```bash
export ATC_MLX_MODEL_PATH=/path/to/whisper-atc-mlx
export ATC_HF_MODEL_PATH=/path/to/whisper-atc-weights
export ATC_MLX_MODEL_V3_PATH=/path/to/whisper-atc-v3-mlx
```
