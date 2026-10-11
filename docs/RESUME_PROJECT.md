# 求职材料：广告 CTR / CVR 多任务预估

## 简历候选版（本人复现并理解后使用）

**广告 CTR / CVR 多任务预估｜DeepCTR-Torch、PyTorch、Criteo**

- 基于 DeepCTR-Torch 搭建 DeepFM、ESMM、SharedBottom、MMoE 统一实验框架；适配 Criteo 真实广告曝光日志，在 164.7 万条固定均匀样本上定义 CTR=`click`、CTCVR=`click∧conversion`，采用严格时间切分和训练期特征统计，排除转化时间、归因结果等事后泄露字段。
- 使用 3 个随机种子、Early Stopping 和最佳检查点完成公平对照：DeepFM CTR AUC 0.7560±0.0002；ESMM/MMoE CTCVR AUC 分别为 0.8524±0.0004、0.8526±0.0003，并同时报告 LogLoss 与点击人群 CVR，避免跨任务误比和只挑单一指标。
- 围绕多任务负迁移完成硬共享、2/3/5 experts 与 CTR:CTCVR 损失权重消融：SharedBottom 相对 ESMM 的 CTCVR/CVR AUC 低 0.0155/0.0173；CTCVR-heavy 使 MMoE CTCVR/CVR AUC 提升 0.0052/0.0072、同时 CTR AUC 降低 0.0071，观察到本实验中任务指标的权衡（不代表统计显著或证明梯度冲突）。实现逐任务训练诊断、数据/检查点 SHA256、重载复算、单元测试与 CI。

## 更短的一行版

基于 DeepCTR-Torch 完成 164.7 万条 Criteo 曝光上的 DeepFM/ESMM/SharedBottom/MMoE CTR-CTCVR 对照，以 3 seeds、时间切分和统一 AUC/LogLoss 研究硬共享、expert 数与任务权重的负迁移/Pareto 权衡，并完成检查点重载审计。

## 口头介绍（约 60 秒）

这个项目不是重新发明模型，而是把广告多任务实验做得定义正确、可复现。我使用 Criteo 官方同时含曝光、点击和转化的数据，固定 CTR 为点击、CTCVR 为点击且转化，CVR 只在点击人群评估；按时间做 70/15/15 切分，预处理只拟合训练期，并排除事后泄露字段。四种模型在 164.7 万条样本上使用相同训练预算和 3 个 seed。结果显示 DeepFM 的 CTR 最好，ESMM 与 MMoE 的 CTCVR 接近；SharedBottom 的部分指标明显更低，这与硬共享可能造成任务干扰的解释一致，但不能单凭该实验确定因果机制。进一步改变 MMoE 的 expert 数和损失权重后，CTR 与转化指标出现稳定的 Pareto 权衡。项目还记录逐 epoch 指标，并用低学习率长耐心实验排除明显的 early-stopping 误停。

## 必须主动说明的边界

- 四个模型主体来自 Apache-2.0 的 DeepCTR-Torch，不应说“从零手写”或“独立提出”。
- 工程搭建、代码、测试、文档与部分调试有 OpenAI Codex 辅助；只有在本人能解释、复现后才应写进简历。
- 主实验是官方全量数据的 10% 均匀样本，不是全量；没有 GPU、线上 A/B、收益或 SOTA 声明。
- DeepFM 只有 CTR 输出，不能拿它与 ESMM/MMoE 的 CTCVR、CVR 做直接优劣比较。

## 投递前自检

只有在你能不看稿回答学习指南中的标签、切分、ESMM 概率链、MMoE gate、AUC/LogLoss、early stopping 和消融控制变量问题，并能亲自重新运行至少一个模型后，再把三条简历描述放入正式简历。
