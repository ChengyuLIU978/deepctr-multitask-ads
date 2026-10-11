# 最终审计

## 结论

**PASS（以“10% 真实数据、CPU 离线求职项目”为明确范围）**。

项目已达到可以写入搜广推算法实习简历的程度：真实公开数据、三个目标模型、三随机种子、Early Stopping、统一评价、两类有控制变量的消融、检查点重载、自动记录、测试、许可与 AI 声明均有证据。PASS 不代表生产上线、全量训练、统计显著性或线上收益已经完成。

## 要求—证据矩阵

| 审计项 | 状态 | 证据 |
|---|---|---|
| 不推倒重做，保留 DeepFM/ESMM/MMoE | PASS | `src/ctr_cvr/models.py`；三个正式运行目录 |
| 真实训练与合理停止 | PASS | 1,646,782 条真实曝光；3 seeds；最大 10 epochs、patience 2，均 best=1、ran=3 |
| 标签定义正确 | PASS | CTR=`click`，CTCVR=`click AND conversion`，CVR clicked-only；测试覆盖 `ctcvr<=click` |
| 时间切分与预处理防泄露 | PASS | 70/15/15 时间边界；同 timestamp 不跨段；训练期 `fit`；泄露字段黑名单 |
| 模型对照 | PASS | `artifacts/final_real_10pct_summary.json`；AUC、LogLoss、耗时、参数、seed 均记录 |
| 指标可比性说明 | PASS | README 与实验日志明确 DeepFM 仅能比较 CTR；ESMM/MMoE 使用相同 CTR/CTCVR 任务 |
| 有意义的消融 | PASS | MMoE 2/3/5 experts；完整特征/仅稀疏；均给研究问题、控制变量和实际结果 |
| 模型保存与加载 | PASS | seed 2026 三个 `*_reload_evaluation.json`，保存指标最大绝对差 0 |
| 可复现与环境 | PASS | 固定 seeds、环境 JSON、数据/检查点 SHA256、PowerShell 脚本、多 seed 汇总校验 |
| 工程测试 | PASS | 8 个 pytest 测试通过；CI 配置为 Windows/Python 3.12 |
| 来源与许可证 | PASS | DeepCTR-Torch Apache-2.0；Criteo 页面标注 CC BY-NC-SA 4.0；第三方声明齐全 |
| AI 辅助和原创边界 | PASS | `docs/AI_DISCLOSURE.md`、README、简历材料均明确说明 |
| GitHub 安全 | PASS | 原始/处理数据、模型权重、环境文件和密钥模式已忽略；可提交 JSON 已移除本机绝对路径 |
| 求职材料与学习材料 | PASS | `docs/RESUME_PROJECT.md`；逐文件学习指南与 30 个追问 |

## 最终数字一致性审计

- 数据量：README、实验日志、数据卡、简历均为 1,646,782 条 10% 样本；未写成全量。
- 切分：所有正式 manifest 均为 1,152,748 / 247,016 / 247,018，边界一致。
- seeds：2026/2027/2028；聚合文件 `run_count=3`。
- 训练轮数：所有正式主实验 `best_epoch=1`、`epochs_ran=3`；文档统一描述为“最多 10、早停到 3”，不是“训练 10 轮”。
- 主指标：README 和简历取自 `final_real_10pct_summary.json` 的均值与样本标准差；没有混入 1% 冒烟或合成 demo。
- 消融：expert=3 复用 seed 2026 正式主实验；expert=2/5 和 sparse-only 有独立运行目录。没有把不同 seed 的数字混为同一控制实验。
- 耗时：文档明确为训练+最佳权重重载+测试预测，不含数据准备；不能解释为线上延迟。
- DeepFM：所有表格的 CTCVR/CVR 留空，不构造不存在的结果。
- MMoE 约束：只说本次违例率为 0，没有声称模型结构保证。

未发现 README、实验 JSON、简历描述和源码任务定义之间的实质矛盾。

## GitHub 发布检查

- `.gitignore` 排除 `data/raw/*`、`data/processed/*`、`.pt/.pth/.ckpt`、`.env*`、缓存与虚拟环境。
- 只白名单小型 JSON 实验证据；检查点仍被全局权重规则排除。
- 下载脚本要求显式 `-AcceptLicense`，项目不重新分发 Criteo 数据。
- 训练/准备/重载生成的路径使用项目相对表示，不发布本机用户名或 home 目录。
- CI 只安装项目依赖并运行单元测试，不需要真实数据或密钥。
- 发布前仍需由仓库所有者选择 GitHub 仓库名称、远端可见性并手动推送；本项目没有替用户创建或发布远端仓库。

## 尚未完成，但不阻止当前求职使用

1. **全量训练**：未完成。当前内存式管线与 16 GB RAM 不适合安全加载 1,646 万行；后续可改为流式 Dataset/DataLoader 后再做。
2. **更强统计结论**：未做 bootstrap 置信区间或显著性检验；不能把 ESMM/MMoE 的微小差距写成显著领先。
3. **线上验证**：没有 A/B、收益、校准曲线或延迟压测；不能写生产上线效果。
4. **GPU 与大规模调参**：未做；当前结论只针对记录的 base 配置与 CPU 环境。
5. **转化延迟处理**：数据观察窗末端可能有标签截尾；未构建成熟窗口或 survival 模型。

## 简历使用门槛

代码和证据层面已可写入简历；个人陈述层面还需要使用者完成学习指南，亲自复现至少一次训练和检查点评估，并能回答标签、时间切分、ESMM 概率链、MMoE gate、AUC/LogLoss 与消融控制变量。不能把 AI 辅助完成但本人无法解释的内容当作独立能力。
