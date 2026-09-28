# 02_makemore

从 bigram 到 MLP 到 WaveNet，重点是**手推反向传播并与 autograd 对拍**。对应规划 Day 2–5 / W1–W2。

## 任务
- Part 1 bigram：计数法 vs 梯度下降+softmax，对比结果
- Part 2 MLP（Bengio 2003）：embedding + hidden + output
- Part 3 训练动力学：初始化、BatchNorm、学习率影响（用 wandb 记录对比实验）
- Part 4 **Backprop Ninja**：关闭 autograd，纯手写整个 MLP + BatchNorm 的反向传播
- Part 5 WaveNet：层级结构

## 验收
- 手推梯度与 PyTorch autograd 最大绝对误差 < 1e-6
- Backprop Ninja 全部手工梯度测试通过（误差 < 1e-6）

## 参考
- https://github.com/karpathy/makemore
