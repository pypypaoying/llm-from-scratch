# 01_micrograd

手写一个自动求导引擎。对应规划 Day 1 / W1。

## 任务
- 实现 `Value` 类：`__add__ / __mul__ / __pow__ / relu / tanh`
- 实现 `backward()`：拓扑排序 + 链式法则 + `_backward` 闭包
- 用自实现的 micrograd 训练一个 2 层 MLP 拟合环形分布，画出决策边界

## 验收
- 不看视频能白板写出 `Value.backward()`
- 能解释「为什么从 `out.grad = 1.0` 开始」「为什么要拓扑排序」

## 参考
- 视频：https://karpathy.ai/zero-to-hero.html （building micrograd）
- 代码：https://github.com/karpathy/micrograd
