# LLMTritonStack

## 项目目录结构

```
LLMTritonStack/
├── configs/              # 配置与资源
├── docs/                 # 设计文档与学习路径
├── scripts/              # 可执行脚本：训练与推理
├── src/llm_lab/          # 主代码包
├── tests/                # 单元测试
├── pyproject.toml        # 项目与依赖定义
├── uv.lock               # 依赖锁文件（建议提交）
└── README.md
```

---

### 根目录文件

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 项目元信息、Python 版本、依赖列表；使用 hatch 构建。 |
| `uv.lock` | 依赖版本锁文件，保证环境一致，建议纳入版本控制。 |
| `.env.example` | 环境变量示例（如需要可复制为 `.env` 使用）。 |
| `.python-version` | 建议的 Python 版本（供 uv/pyenv 等使用）。 |

---

### `configs/` — 配置与模型资源

| 路径 | 说明 |
|------|------|
| `configs/model/config.py` | 模型相关配置示例（可与 `src/llm_lab/config.py` 对照）。 |
| `configs/model/tokenizer.json` | Tokenizer 词表与 BPE 等配置（HuggingFace 格式）。 |
| `configs/model/tokenizer_config.json` | Tokenizer 行为配置（特殊 token、chat 模板等）。 |

训练与推理脚本中的 `--tokenizer_path` 默认指向 `./configs/model`（目录），即使用上述 tokenizer 文件。

---

### `docs/` — 文档与学习路径

| 文件 | 说明 |
|------|------|
| `design.md` | 项目整体设计：愿景、目录约定、后端抽象、阶段规划（阶段 0～5）等。 |
| `DIYModel_TODO.md` | 从零完善 DIY 模型的任务列表（lm_head、训练循环、KV cache、配置保存等）。 |
| `DIYModel_advanced_path.md` | 进阶学习路径：RoPE、规模调整、预训练/SFT、工程化等。 |
| `learning_plan_short_term.md` | 短期学习计划：Triton 算子、后端整合、MoE/RL 拓展。 |

---

### `scripts/` — 可执行脚本

| 脚本 | 说明 |
|------|------|
| `pretrain.py` | 预训练入口。读 jsonl、用 `TritonMindForCausalLM` + `SimplePretrainDataset`，支持 AMP、梯度累积、余弦学习率、按步保存。 |
| `sft_train.py` | 有监督微调。从 `--from_checkpoint` 加载预训练权重，在 SFT 数据上训练，支持 AMP、梯度累积。 |
| `infer.py` | 交互式推理。加载 checkpoint 与 tokenizer，支持 KV cache 增量解码，循环读入 prompt 并输出生成结果。 |

三个脚本均支持 `--tokenizer_path`（默认 `./configs/model`）；`infer.py` 会从 checkpoint 的 config 中过滤出模型结构字段以兼容旧版保存的配置。

---

### `src/llm_lab/` — 主代码包

核心包名：`llm_lab`。对外主要暴露：`TritonMindConfig`、`TritonMindForCausalLM`、数据集类、RoPE 工具、logger。

| 文件/目录 | 说明 |
|-----------|------|
| `config.py` | **TritonMindConfig**：模型结构参数（vocab_size、hidden_size、num_layers、num_heads、RoPE、train_max_length 等），不含训练超参。 |
| `__init__.py` | 导出 Config、Model、ForCausalLM、数据集、RoPE、logger 等，供脚本和外部调用。 |

---

#### `src/llm_lab/core/` — 模型结构

只定义网络结构和数据流，不写死具体算子实现；底层算子通过 `ops/` 或后续 backend 接入。

| 文件 | 说明 |
|------|------|
| `model.py` | **TritonMindModel**：embed → N × ModelBlock → RMSNorm，输出 hidden_states；支持 `past_key_values` / `use_cache` 做增量解码。**TritonMindForCausalLM**：在 Model 之上加 lm_head，输出 logits，用于训练与生成。 |
| `transformer_block.py` | **ModelBlock**：单层 Transformer（Attention + MLP，含残差与 RMSNorm）。 |
| `attention.py` | 多头注意力：Q/K/V 线性变换、RoPE、缩放点积注意力、输出投影。 |
| `mlp.py` | 前馈子层：gate/up 线性 + 激活 + down 投影（SwiGLU 等）。 |

---

#### `src/llm_lab/ops/` — 可插拔算子（含 autograd）

当前模型实际使用的算子实现与自定义 backward 的封装点。

| 文件 | 说明 |
|------|------|
| `rmsnorm.py` | **RMSNorm**：对最后一维做 RMS 归一化，带可学习 scale；被 core 的 Model 与 Block 使用。 |
| `linear.py` | 线性层封装（若需对接自定义 kernel，可在此接 `torch.autograd.Function`）。 |

后续 Triton/CUDA 的 backward 也可在此以 `Function` 形式挂接。

---

#### `src/llm_lab/backends/` — 后端实现（Torch / Triton / CUDA）

按后端分目录存放「同一接口」的算子实现，便于通过 registry 切换。

| 路径 | 说明 |
|------|------|
| `registry.py` | 后端注册与选择逻辑（当前可为空，预留按配置切换 Torch/Triton）。 |
| `torch/` | PyTorch 参考实现：`linear.py`、`rmsnorm.py`、`softmax.py`（当前 core 直接使用 `ops/`，此处可作对照或日后接入 registry）。 |
| `triton/` | Triton kernel 实现：`linear.py`、`rmsnorm.py`、`softmax.py`（待实现，用于阶段 1～2 的替换与对比）。 |
| `cuda/` | 手写 CUDA 实现（可选，长期）。 |

---

#### `src/llm_lab/datasets/` — 数据集

| 文件 | 说明 |
|------|------|
| `datasets.py` | **SimplePretrainDataset**：读 jsonl，按 tokenizer 编码、截断/padding，返回 input_ids、labels、loss_mask。**SimpleSFTDataset**：SFT 用对话格式，同样产出 input_ids/labels/loss_mask。**PretrainDataset / SFTDataset**：可扩展的迭代式或更复杂版本。 |

---

#### `src/llm_lab/utils/` — 工具

| 文件 | 说明 |
|------|------|
| `rope.py` | **precompute_freqs_cis**：预计算 RoPE 的 cos/sin；**apply_rotary_pos_emb**：对 Q/K 应用旋转位置编码。 |
| `logger.py` | 统一 logger：`setup_logger`、`get_logger`，支持按脚本自动写日志到 `logs/`。 |

---

#### `src/llm_lab/benchmark/` — 性能测试

| 文件 | 说明 |
|------|------|
| `kernel_bench.py` | 单 kernel 级 benchmark（如 RMSNorm、Softmax、Linear 的 Torch vs Triton）。 |
| `layer_bench.py` | 单层（如 Attention、MLP）的耗时/带宽测试。 |
| `model_bench.py` | 整模型前向/训练步的吞吐（如 token/s、显存）。 |

---

#### `src/llm_lab/profiling/` — 性能分析

| 文件 | 说明 |
|------|------|
| `profiler.py` | 与 PyTorch profiler 等对接的封装。 |
| `roofline.py` | Roofline 分析辅助：FLOPs、内存流量等计算，用于评估 kernel 瓶颈。 |

---

#### `src/llm_lab/training/` — 训练组件

| 文件 | 说明 |
|------|------|
| `trainer.py` | 通用训练循环封装（当前脚本仍以自写循环为主，此处可复用或扩展）。 |
| `optimizer.py` | 优化器或学习率调度相关工具。 |

---

### `tests/` — 单元测试

| 文件 | 说明 |
|------|------|
| `test_attention.py` | Attention 模块的数值或形状测试。 |
| `test_linear.py` | 线性层相关测试。 |
| `test_rmsnorm.py` | RMSNorm 的数值与梯度测试。 |

---

## 运行与数据

- **环境**：建议 Python 3.12+，使用 `uv sync` 安装依赖。
- **预训练数据**：jsonl，每行一个 JSON（如 `{"text": "..."}`），脚本会按 `text` 字段用 tokenizer 编码。
- **SFT 数据**：格式与脚本约定一致（见 `datasets/datasets.py` 或脚本帮助）。
- **Checkpoint**：保存为 `{"config": ..., "state_dict": ...}`，推理时会自动过滤 config 中模型结构字段以兼容旧版。

更多阶段规划与学习路径见 `docs/design.md` 与 `docs/learning_plan_short_term.md`。
