# 求职材料：广告 CTR / CVR 多任务预估

## 简历版（建议直接使用）

**广告 CTR / CVR 多任务预估｜DeepCTR-Torch、PyTorch、Criteo**

- 基于 DeepCTR-Torch 搭建 DeepFM、ESMM、MMoE 统一实验框架；适配 Criteo 真实广告曝光日志，在 164.7 万条固定均匀样本上定义 CTR=`click`、CTCVR=`click∧conversion`，采用严格时间切分和训练期特征统计，排除转化时间、归因结果等事后泄露字段。
- 使用 3 个随机种子、Early Stopping 和最佳检查点完成公平对照：DeepFM CTR AUC 0.7560±0.0002；ESMM/MMoE CTCVR AUC 分别为 0.8524±0.0004、0.8526±0.0003，并同时报告 LogLoss 与点击人群 CVR，避免跨任务误比和只挑单一指标。
- 完成连续特征组和 MMoE 2/3/5 experts 消融；发现连续特征使三模型 CTR AUC 提升 0.071–0.080，而更多 experts 只改善部分 CTCVR 指标。实现数据/检查点 SHA256、模型重载复算、实验清单、单元测试与 CI，保证结果可追溯。

## 更短的一行版

基于 DeepCTR-Torch 完成 164.7 万条 Criteo 曝光上的 DeepFM/ESMM/MMoE CTR-CTCVR 对照，使用时间切分、3 seeds、Early Stopping、统一 AUC/LogLoss、两类消融及检查点重载审计；ESMM/MMoE CTCVR AUC 分别达到 0.8524±0.0004/0.8526±0.0003。

## 口头介绍（约 60 秒）

这个项目不是重新发明模型，而是把一个广告多任务实验做得定义正确、可复现。我使用 Criteo 官方同时含曝光、点击和转化的数据，固定 CTR 为点击、CTCVR 为点击且转化，CVR 只在点击人群评估。为了模拟线上用过去预测未来，按时间做 70/15/15 切分，预处理只拟合训练期，并排除了转化时间、归因结果等泄露字段。模型使用 DeepCTR-Torch 的 DeepFM、ESMM 和 MMoE，在 164.7 万条样本上跑 3 个 seed 和 early stopping。结果显示 DeepFM 的 CTR 最好；ESMM 与 MMoE 的 CTCVR 很接近，不能宣称显著胜出。我还做了连续特征和 expert 数量消融，并通过数据指纹、最佳检查点重载和测试保证结果能复现。

## 必须主动说明的边界

- 三个模型主体来自 Apache-2.0 的 DeepCTR-Torch，不应说“从零手写”或“独立提出”。
- 工程搭建、代码、测试、文档与部分调试有 OpenAI Codex 辅助；只有在本人能解释、复现后才应写进简历。
- 主实验是官方全量数据的 10% 均匀样本，不是全量；没有 GPU、线上 A/B、收益或 SOTA 声明。
- DeepFM 只有 CTR 输出，不能拿它与 ESMM/MMoE 的 CTCVR、CVR 做直接优劣比较。

## 投递前自检

只有在你能不看稿回答学习指南中的标签、切分、ESMM 概率链、MMoE gate、AUC/LogLoss、early stopping 和消融控制变量问题，并能亲自重新运行至少一个模型后，再把三条简历描述放入正式简历。
