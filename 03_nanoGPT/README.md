# 03_nanoGPT

从零手写一个完整的 Transformer / GPT，在 TinyStories 上训练与生成。对应规划 Day 6 / W2–W3。

## 任务
- 禁止使用 `nn.MultiheadAttention`、禁止抄 nanoGPT，自己写：
  `RMSNorm` → `CausalSelfAttention`（含 mask、多头 reshape）→ `MLP`(SwiGLU 可选)
  → `Block`（Pre-Norm + 残差）→ `GPT`（token emb + position emb + N×Block + LM head + weight tying）
- 数据管道：TinyStories（`roneneldan/TinyStories`），字符级或简易 BPE，实现 `get_batch()`
- 训练：`d_model=128, n_layer=4, n_head=4, ctx=256, batch=8, bf16 + grad accum 4`（4060 8GB 内可跑）
- `generate()`：temperature + top-k 采样
- W3 加入 wandb 日志、checkpoint、grad clipping、cosine LR、断点续训

## 验收
- 代码跑通无报错；loss 从 ≈10 降到 < 3；能生成语法基本通顺的英文小故事
- 能对着自己的代码逐行说出每个 tensor 的 shape 变化

## 参考（卡住超过 30 分钟才看）
- https://github.com/karpathy/nanoGPT
- https://github.com/karpathy/ng-video-lecture
