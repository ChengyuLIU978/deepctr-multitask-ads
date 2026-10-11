# 广告 CTR / CVR 多任务预估（DeepCTR-Torch）

面向搜广推算法实习的可复现项目：在同一份真实曝光日志、同一时间切分和同一特征合同下，对比 DeepFM、ESMM 与 MMoE。模型主体调用成熟开源 DeepCTR-Torch；本项目完成数据适配、任务定义、防泄露、训练控制、统一评估、消融、检查点重载与实验审计，不把第三方实现包装成从零原创。

## 已验证结论

- 官方 Criteo 原始数据共 16,468,027 条曝光；主实验采用固定 seed 的 10% 均匀样本，共 1,646,782 条，不按标签过采样。
- 按曝光时间做 70% / 15% / 15% 切分：训练 1,152,748、验证 247,016、测试 247,018；相同时间戳不会跨集合。
- 主实验为 CPU、3 个随机种子、最多 10 epochs、patience=2 的 early stopping。三种模型都在第 1 epoch 达到最优验证损失，并在第 3 epoch 停止；不是人为只跑 1 epoch。
- 10% 数据是当前 16 GB 内存、CPU 环境中安全完成全部多 seed 与消融的最大已验证规模。全量数据没有完成训练，因此没有任何“全量结果”声明。
- seed 2026 的三个最佳检查点均已从磁盘重新加载，测试指标与训练记录逐项一致，最大绝对差为 0。

主结果为 3 个 seed 的均值 ± 样本标准差：

| 模型 | CTR AUC | CTR LogLoss | CTCVR AUC | CTCVR LogLoss | CVR AUC（clicked） | CVR LogLoss（clicked） | 单次耗时（秒） |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepFM | 0.7560 ± 0.0002 | 0.5592 ± 0.0015 | — | — | — | — | 49.49 ± 0.02 |
| ESMM | 0.7538 ± 0.0002 | 0.5613 ± 0.0006 | 0.8524 ± 0.0004 | 0.1512 ± 0.0001 | 0.8283 ± 0.0012 | 0.3031 ± 0.0008 | 45.77 ± 0.80 |
| MMoE | 0.7509 ± 0.0007 | 0.5656 ± 0.0010 | 0.8526 ± 0.0003 | 0.1511 ± 0.0003 | 0.8247 ± 0.0011 | 0.3043 ± 0.0011 | 50.86 ± 2.88 |

DeepFM 只输出 CTR，因此只能与其他模型的 CTR 指标直接比较，不能拿它和 ESMM/MMoE 的 CTCVR 或 CVR 指标比较。ESMM 与 MMoE 的 CTCVR 差异很小；在本实验里不能据此宣称某个多任务模型显著胜出。完整逐 seed 数字和限制见 [实验日志](docs/EXPERIMENT_LOG.md)。

## 标签与评价口径

| 名称 | 定义 | 训练/评估人群 |
|---|---|---|
| 曝光 | 数据中的每一行 | 全部样本 |
| CTR | `click` | 全部曝光 |
| CTCVR | `click AND conversion` | 全部曝光 |
| CVR | 点击后的转化 | 只在 `click == 1` 的样本上评估 |

ESMM 输出 `P(CTR)` 与 `P(CTCVR)=P(CTR)×P(CVR)`，结构上保证 `P(CTCVR)≤P(CTR)`。MMoE 的两个任务也固定为 CTR/CTCVR，使监督信号与 ESMM 一致；其输出头相互独立，所以额外记录概率包含关系违例率。主实验中观测值为 0，但这不是 MMoE 的结构保证。

测试集实际 CTR 为 0.36520、CTCVR 为 0.04963；90,212 条点击曝光上的 CVR 为 0.13589。原始数据中 806,196 个转化全部同时有点击；代码仍用交集构造 CTCVR，以把任务合同写死并防止换数据时悄悄改变语义。

## 防泄露设计

- 先按 `timestamp` 稳定排序，再做时间切分；训练早于验证，验证早于测试。
- 连续特征的均值、标准差与时间原点只在训练段拟合；验证/测试只调用保存的转换器。
- 类别字段采用固定哈希桶，既控制高基数 embedding 内存，也避免用未来类别构建词表。
- 禁用事后字段：`conversion_timestamp`、`conversion_id`、`attribution`、`click_pos`、`click_nb`、`cpo`、`time_since_last_click`。
- 训练与重载评估都会核验准备后数据的 SHA256；多 seed 汇总还会检查数据指纹、时间边界、特征集、模型容量和训练预算。

## 有意义的消融

### 1. MMoE expert 数量

研究问题：更多 expert 是否会在相同数据、特征、容量、seed 与训练预算下稳定改善两个任务？

| Experts | 参数量 | CTR AUC | CTCVR AUC | CVR AUC（clicked） | 耗时（秒） |
|---:|---:|---:|---:|---:|---:|
| 2 | 2,660,282 | 0.7475 | 0.8520 | 0.8225 | 46.74 |
| 3 | 2,668,346 | **0.7517** | 0.8524 | **0.8236** | 54.17 |
| 5 | 2,684,474 | 0.7484 | **0.8543** | 0.8216 | 54.66 |

5 experts 的 CTCVR 略好，但 CTR、clicked-CVR 与校准更差；3 experts 更均衡。结论是“expert 更多不等于所有目标都更好”，不是挑单个最好数字。

### 2. 特征组

研究问题：曝光时可用的 `cost_log`、周期时间与相对时间连续特征是否提供了超出匿名离散 ID 的信息？控制模型、seed、切分与训练预算，只去掉这组连续特征。

| 模型 | 完整特征 CTR AUC | 仅稀疏 CTR AUC | 完整特征 CTCVR AUC | 仅稀疏 CTCVR AUC |
|---|---:|---:|---:|---:|
| DeepFM | 0.7558 | 0.6847 | — | — |
| ESMM | 0.7535 | 0.6764 | 0.8521 | 0.8228 |
| MMoE | 0.7517 | 0.6717 | 0.8524 | 0.8237 |

完整特征显著改善 CTR 与 CTCVR 排序；clicked-CVR 的改善较小，因此不能写成“所有任务同幅度提升”。

## 复现

Windows PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\download_criteo.ps1 -AcceptLicense
.\scripts\prepare_real_data.ps1
.\scripts\run_main_experiments.ps1
.\scripts\run_ablations.ps1
.\scripts\evaluate_checkpoints.ps1
```

`prepare_real_data.ps1` 默认准备 10% 样本。下载脚本只访问 Criteo 官方 Hugging Face 组织，要求显式接受数据许可，并核验原始压缩文件 SHA256。原始与处理后数据、检查点均由 `.gitignore` 排除。

`requirements.txt`/`pyproject.toml` 给出支持范围；`requirements-lock.txt` 固化本次 Windows CPU 实验的完整依赖版本。若要精确复刻环境，可先安装 lock，再用 `python -m pip install -e . --no-deps` 安装本项目。

快速验证代码闭环而不下载数据：

```powershell
.\scripts\run_demo.ps1
.\.venv\Scripts\python -m pytest
```

合成 demo 只证明工程可运行，任何 demo 指标都不得写进简历。

## 工程产物

每次训练写入 `artifacts/<run>/`：

- `data_manifest.json`：数据指纹、标签、时间边界、分区基准率和训练期预处理统计；
- `environment.json`：实际 Python、PyTorch、DeepCTR-Torch 与数据栈版本；
- `<model>_result.json`：训练历史、最佳 epoch、参数量、耗时、检查点哈希和测试指标；
- `run_manifest.json`：整次运行配置与结果；
- `<model>_reload_evaluation.json`：重新加载检查点后的独立复算证据。

仓库中只保留小型 JSON 实验证据；大型数据与 `.pt` 检查点不提交。工程审查见 [最终审计](docs/FINAL_AUDIT.md)，学习路线与 30 个追问见 [源码学习指南](docs/LEARNING_GUIDE.md)，简历版本见 [求职材料](docs/RESUME_PROJECT.md)。

## 项目结构

```text
src/ctr_cvr/          数据、模型工厂、训练、重载评估、汇总
scripts/              环境、下载、准备、主实验、消融、重载检查
tests/                标签、切分、预处理与模型配置不变量
artifacts/            可提交的 JSON 证据；本地权重被忽略
docs/                 数据卡、实验日志、学习指南、简历与最终审计
.github/workflows/    Windows CPU 单元测试
```

## 来源、许可与诚实边界

- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)，Apache-2.0；模型主体来自该依赖。
- [Criteo Attribution Modeling for Bidding Dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)，发布页标注 CC BY-NC-SA 4.0；仅用于个人学习/非商业研究。
- [数据论文](https://arxiv.org/abs/1707.06409)、[ESMM 论文](https://dl.acm.org/doi/10.1145/3209978.3210104)、[MMoE 论文](https://dl.acm.org/doi/10.1145/3219819.3220007)。

本仓库自有代码采用 MIT，第三方软件与数据仍受各自许可约束。项目结构、代码、测试、文档和部分调试由 OpenAI Codex 辅助完成；使用者必须亲自理解并复现实验。不得声称独立提出/从零手写三个模型，也不得声称完成全量数据或线上 A/B 测试。详情见 [第三方声明](THIRD_PARTY_NOTICES.md) 与 [AI 辅助声明](docs/AI_DISCLOSURE.md)。
