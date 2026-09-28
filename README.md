# llm-from-scratch

> 一个信息工程出身、推免至哈工大（深圳）的学生，在研 0 期间「从零手写大模型地基」的全过程记录。
> 起点：2026-09-28 ｜ 主线资源：Andrej Karpathy《Neural Networks: Zero to Hero》+ Stanford CS336

## 这个仓库是什么

不是教程搬运，是**我自己一行一行敲出来、跑通、并能讲清每个 tensor shape 的实现**。
核心目标只有一个：把「看视频都懂、一写就废」变成「能在白板上默写 Transformer」。

## 研 0 四大目标（全局）

1. 夯实基础能力（手写自动求导 → 手写 Transformer）
2. 快速过一遍 CS231n
3. 认真学 CS336《Language Modeling from Scratch》
4. 完成 Hello-Agents，并拿到一段广州的大模型/Agent 实习

本仓库对应的是**第 1 项 + 第 3 项的 A1/A2 地基部分**。

## 阶段甘特（2026.10 手写地基月）

```
W1 (09.29–10.05)  micrograd + makemore 1–5   → 见 01_micrograd / 02_makemore
W2 (10.06–10.12)  WaveNet + 从零手写 GPT      → 见 02_makemore / 03_nanoGPT
W3 (10.13–10.19)  复现 nanoGPT + 训练工程     → 见 03_nanoGPT
W4 (10.20–10.31)  巩固 + 博客 + 面试手撕模板  → 见 04_interview_handwrite / blog
```

## 目录结构

| 目录 | 内容 | 对应规划 |
|---|---|---|
| `01_micrograd/` | 手写自动求导引擎（Value 类 + 拓扑排序 backward） | Day 1 / W1 |
| `02_makemore/` | bigram → MLP → BatchNorm → WaveNet，含手推反向传播对拍 | Day 2–5 / W1–W2 |
| `03_nanoGPT/` | 从零手写 Transformer / GPT，TinyStories 训练与生成 | Day 6 / W2–W3 |
| `04_interview_handwrite/` | 面试手撕模板：MHA、LayerNorm/RMSNorm、BPE、top-k sampling | W4 |
| `notes/` | 每周周记与概念笔记（拓扑排序、Pre-LN vs Post-LN 等） | 全程 |
| `blog/` | 对外发布的技术博客草稿 | W4 |
| `data/` | 数据集存放（已 gitignore，不上传） | — |

## 进度清单（Day 1–7 验收）

- [ ] micrograd 自实现版跑通，能白板写出 `backward()`
- [ ] MLP 反向传播手推，与 autograd 对拍误差 < 1e-6
- [ ] Backprop Ninja：关掉 autograd 全手写梯度，全部测试通过
- [ ] BatchNorm / LayerNorm / RMSNorm 区别能讲清，Pre-LN vs Post-LN 能讲清
- [ ] Attention 手算与代码一致（误差 < 1e-5）
- [ ] 从零手写 Transformer 在 TinyStories 上 loss < 3 且能生成通顺故事
- [ ] 发布 1 篇博客《从零手写 Transformer 全过程》

## 环境

- 本地：RTX 4060 Laptop 8GB
- conda 环境 `llm`：Python 3.11 + PyTorch 2.6.0+cu124
- 激活：`conda activate llm`

## 三条纪律

1. 不追求看完，只追求写出来——所有代码自己敲，禁止复制粘贴。
2. 卡住 30 分钟就换策略（查文档 → 搜 issue → 打印 shape → 缩规模 → 最后才看题解）。
3. 每天都要有产出落到 Git 上，没有 commit 的一天等于没学。
