# 短期学习计划：Triton → 模型架构（MoE / RL）

**前提**：已完成项目迁移，DIYModel 基础与进阶（RoPE、预训练、SFT、KV cache、配置与 checkpoint）均已跑通。

**目标**：在现有 Torch baseline 之上，学习 Triton 内核替换与验证流程，并延伸到模型层面的 MoE、强化学习（RLHF/DPO）等方向。

**对应 design.md**：阶段 0 已完成 → 本计划覆盖**阶段 1～2 为核心**，并预留阶段 3～5 及 MoE/RL 的衔接。

---

## 学习路径总览

```
阶段 A：Triton 基础算子（约 2 周）     ← 对应 design 阶段 1
  └─ RMSNorm / Softmax / Linear(naive) + kernel benchmark + analysis.md

阶段 B：后端整合与梯度验证（约 1～1.5 周）  ← 对应 design 阶段 2
  └─ registry 切换、autograd backward、gradcheck / 短步训练对比

阶段 C：模型层拓展（约 2～3 周，可与 B 并行或之后）
  └─ MoE 架构入门 或 强化学习（DPO/RLHF）入门
```

**建议**：先完成 A → B，再选 C 中一个方向深入；若时间紧，C 中 MoE 与 RL 二选一即可。

---

## 阶段 A：Triton 基础算子（约 2 周）

**目标**：掌握 Triton 的 block-level 编程模型，实现三个基础算子的 Triton 版本，并能用现有 `benchmark/` 做对比与简单分析。

### A.1 环境与文档（1 天）

- [ ] 确认本机/环境已安装 Triton，且与当前 PyTorch/CUDA 兼容。
- [ ] 通读 [Triton 官方 Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html) 前几章（如 01-vector-add、02-fused-softmax），理解：
  - `@triton.jit`、`tl.load`/`tl.store`、grid、BLOCK 含义。
  - 为何要做 2D grid、如何用 `num_warps` 等。
- [ ] 在 `docs/` 下新建 `triton_notes.md`，记录：block 与 thread 的对应关系、常用 pattern（reduce、element-wise）。

### A.2 RMSNorm（3～4 天）

- [ ] 阅读 `backends/torch/rmsnorm.py`，明确输入输出与数学形式（方差、eps、归一化）。
- [ ] 在 `backends/triton/rmsnorm.py` 实现 Triton 版 RMSNorm（forward 即可）：
  - 按 `BLOCK_SIZE` 分块，每块内先算平方和再开方，再逐元素缩放。
  - 与 Torch 实现做 `torch.allclose` 数值对比（多种 shape：长序列、大 hidden_size）。
- [ ] 在 `benchmark/kernel_bench.py`（或单独脚本）中增加 RMSNorm 的 kernel 级对比（Torch vs Triton），记录带宽/延迟。
- [ ] 在 `experiments/` 下建 `rmsnorm/`，写简短 `analysis.md`：shape 对性能的影响、何时 Triton 更优。

### A.3 Softmax（3～4 天）

- [ ] 阅读 `backends/torch/softmax.py`，理解当前接口（last dim softmax）。
- [ ] 实现 Triton Softmax（建议参考官方 fused-softmax 教程）：
  - 使用 online softmax 或分块 max+exp+sum，避免数值溢出。
  - 同样做数值对比与 kernel benchmark。
- [ ] 在 `experiments/softmax/` 写 `analysis.md`：与 Torch 的差异、适用场景。

### A.4 Linear（naive）（3～4 天）

- [ ] 阅读 `backends/torch/linear.py` 的调用方式（shape、是否 bias）。
- [ ] 实现一个 **naive** Triton MatMul（不追求极致优化）：
  - 2D grid，每个 block 负责输出一块 C = A @ B。
  - 与 Torch `F.linear` 数值对比，再上 kernel benchmark。
- [ ] 在 `experiments/linear_naive/` 写 `analysis.md`：内存访问模式、与 cuBLAS 的差距（仅作感性认识）。

### 阶段 A 交付物

- `backends/triton/` 下：`rmsnorm.py`、`softmax.py`、`linear.py` 可运行且与 Torch 数值一致。
- `benchmark/` 能跑出 Triton vs Torch 的 kernel 级报告（至少 RMSNorm/Softmax/Linear 各一）。
- `experiments/{rmsnorm,softmax,linear_naive}/analysis.md` 各一份。

---

## 阶段 B：后端整合与梯度验证（约 1～1.5 周）

**目标**：把 Triton 算子接到当前模型上，通过 registry 切换后端，并验证梯度正确性与短步训练一致性。

### B.1 Registry 与模型接入（2～3 天）

- [ ] 阅读并完善 `backends/registry.py`：支持按配置或环境变量选择 `torch` / `triton`。
- [ ] 确保 `core/` 中（如 `transformer_block`、`attention`、`mlp`）通过 registry 获取 RMSNorm、Softmax、Linear 的实现，而不是直接 `import torch` 或写死实现。
- [ ] 在仅使用 Triton RMSNorm（其余仍 Torch）的情况下，跑通 `scripts/pretrain.py` 若干 step，确认无报错。

### B.2 Backward 与 gradcheck（3～4 天）

- [ ] 为 Triton RMSNorm 实现 `torch.autograd.Function`，在 `ops/` 中封装 forward/backward，backward 可用 Triton 或先退回 Torch 公式实现。
- [ ] 使用 `torch.autograd.gradcheck` 对 Triton RMSNorm（及后续 Softmax/Linear）做梯度检查。
- [ ] 若 Triton Softmax/Linear 需要自定义 backward，同样在 `ops/` 中实现并 gradcheck。

### B.3 短步训练对比（1～2 天）

- [ ] 固定 seed、相同数据与超参，分别用 `torch` 后端和 `triton` 后端各跑 50～100 step 预训练。
- [ ] 对比 loss 曲线是否接近（允许小幅数值差异）；若有明显偏差，回到 gradcheck 与数值对比排查。

### 阶段 B 交付物

- 通过 registry 可切换 Torch / Triton 后端，且当前脚本（如 pretrain）无需改核心逻辑。
- 所有替换过的 Triton 算子通过 gradcheck；短步训练 loss 与 Torch baseline 基本一致。
- 在 `docs/` 或 `experiments/` 中简短记录「如何切换后端、如何跑 gradcheck」。

---

## 阶段 C：模型层拓展（约 2～3 周，选做或二选一）

在完成 A、B 后，可根据兴趣选一个方向先做，避免同时铺开太多。

---

### C1. MoE 架构入门

**目标**：理解稀疏 MoE 的路由与计算形式，在当前仓库里实现一个「最小可用的 MoE 层」并跑通前向。

- [ ] **阅读**：MoE 综述或 Mixtral/DeepSeek-MoE 中与 MoE 相关的部分，搞清：
  - Router（top-k、load balance loss）、Expert（FFN）、token 与 expert 的对应关系。
- [ ] **实现**：在 `core/` 下新增 `moe.py`（或 `moe_layer.py`）：
  - 若干 FFN 作为 expert，一个 router（线性层 + softmax），对每个 token 选 top-k expert，加权或拼接输出。
  - 先不追求 Triton 优化，用 Torch 实现即可；接口与现有 `MLP` 类似（输入/输出 shape 一致），便于插到一层替换。
- [ ] **验证**：在 1 个 block 里用 MoE 替换原有 MLP，跑 1 个 batch 前向 + backward，loss 能下降即可。
- [ ] **文档**：在 `docs/` 写 `moe_notes.md`，记录：路由方式、top-k 与负载均衡、与 design 阶段 5 的「MoE kernel 实验」的衔接思路。

---

### C2. 强化学习（DPO / RLHF）入门

**目标**：理解偏好对齐的基本思路，实现一个最小可用的 DPO 或简化版 RLHF 训练流程。

- [ ] **阅读**：
  - DPO：Direct Preference Optimization 论文或博客（无显式 reward model、直接从 preference 数据优化）。
  - 可选：InstructGPT/RLHF 中 PPO 的四个模型（policy、ref、critic、reward）的角色。
- [ ] **数据**：准备或构造「偏好对」数据格式（prompt、chosen、rejected），可先用手工构造的极小数据集。
- [ ] **实现**：在 `scripts/` 下新增 `dpo_train.py`（或 `rlhf_mini.py`）：
  - 读入 SFT checkpoint，加载为 policy；reference 可为同一 checkpoint 的拷贝（不更新）或 EMA。
  - 实现 DPO loss（或简化版 preference loss），对 policy 做若干 step 更新。
- [ ] **验证**：过拟合几对 preference 数据，确认 loss 下降、更新的是 policy 而非 ref。
- [ ] **文档**：在 `docs/` 写 `rl_notes.md`，记录：DPO 与 PPO 的区别、当前脚本的假设与局限、与 design 阶段 5「RLHF kernel 实验」的衔接。

---

## 时间与优先级建议

| 周期       | 建议内容 |
|------------|----------|
| 第 1～2 周 | 阶段 A：Triton 三个算子 + benchmark + analysis |
| 第 3 周    | 阶段 B：registry、backward、gradcheck、短步训练 |
| 第 4～5 周 | 阶段 C：MoE **或** RL 选一个做到底 |
| 第 6 周    | 收尾：文档整理、跑一次完整 pretrain 对比、写小结 |

若时间紧张，可只做 **A + B**，C 留作下一轮计划；这样已经完成 design 中的阶段 1～2，为后续「进阶优化与 Attention（阶段 3～4）」打好基础。

---

## 与 design.md 阶段的对应关系

| 本计划阶段 | design.md 阶段 | 交付物概要 |
|------------|----------------|------------|
| A          | 阶段 1         | Triton RMSNorm/Softmax/Linear(naive) + kernel benchmark + analysis.md |
| B          | 阶段 2         | registry 切换、backward、gradcheck、短步训练一致 |
| C1         | 阶段 5 部分    | MoE 层实现与笔记，为后续 MoE kernel 实验打基础 |
| C2         | 阶段 5 部分    | DPO/RLHF 最小流程与笔记，为后续 RLHF kernel 实验打基础 |

完成 A+B 后，可自然过渡到 design 的**阶段 3**（进阶 Triton 优化、fused MLP、Roofline）和**阶段 4**（Triton Attention、KV cache、模型级 benchmark）。

---

## 推荐资料

- **Triton**：[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- **MoE**：Mixtral 论文、DeepSeek-MoE / 任意一篇 MoE 架构说明
- **RLHF/DPO**：InstructGPT、DPO 论文；Hugging Face TRL 文档（可选，作参考）

如有需要，可以把「阶段 A 的 Triton RMSNorm 模板代码」或「registry 与 core 的接入示例」单独拆成小任务，按步骤实现。
