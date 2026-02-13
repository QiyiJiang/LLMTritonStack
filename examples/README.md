# Examples / 学习与教程代码

本目录用于存放**学习过程中写的示例与教程代码**，与正式入库的 `src/llm_lab/`、`scripts/` 区分开。

## 约定

- **不要求**通过包导入：可直接 `uv run examples/xxx/yyy.py` 或 `python examples/xxx/yyy.py` 运行。
- 按主题分子目录（如 `triton/`、`cuda/`），同一教程的多个示例可用数字前缀排序（如 `01_vector_add.py`、`02_fused_softmax.py`）。
- 可与 `docs/` 中的学习计划、教程笔记对应；若某示例日后稳定、要复用到项目里，再考虑迁入 `src/llm_lab/` 或 `scripts/`。

## 子目录说明

| 子目录 | 用途 |
|--------|------|
| `triton/` | Triton 官方教程或自练 kernel（如 vector add、softmax、matmul）。 |

## 注意

- 不要用 `tmp/` 存学习代码：`tmp/` 已在 `.gitignore` 中，其中的文件不会被提交。
- 希望保留并随仓库版本管理的学习代码请放在本目录下。
