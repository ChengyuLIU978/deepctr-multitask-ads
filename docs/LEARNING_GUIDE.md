# 源码学习与面试指南

目标不是背诵文件，而是能从一条曝光讲到标签、切分、特征、三种模型、损失、指标和实验结论，并能明确哪些代码来自开源库。

## 一张执行地图

```text
官方 .tsv.gz
  -> prepare_criteo.py：分块读取、固定 seed 均匀采样、来源哈希
  -> data.py：校验标签、构造 ctcvr、时间切分、训练期拟合特征
  -> models.py：用 DeepCTR-Torch 创建 DeepFM / ESMM / MMoE
  -> train.py：early stopping、最佳检查点、测试评估、实验清单
  -> evaluate.py：重建预处理和模型、重载权重、复算测试指标
  -> summarize.py：校验可比性、汇总多 seed 均值与样本标准差
```

## 逐文件阅读

| 文件 | 职责 | 必须理解的入口 |
|---|---|---|
| `src/ctr_cvr/prepare_criteo.py` | 大文件分块读取、无标签均匀采样、原始/处理文件哈希和计数 | `main` |
| `src/ctr_cvr/data.py` | schema、标签合同、时间切分、特征拟合/变换、manifest 恢复 | `validate_and_label`、`temporal_split`、`FeaturePreprocessor` |
| `src/ctr_cvr/models.py` | DeepCTR-Torch 的薄封装、容量和 expert 数配置 | `build_model`、`parameter_count` |
| `src/ctr_cvr/metrics.py` | CTR、CTCVR 和 clicked-CVR 的统一评价 | `evaluate_ctr`、`evaluate_multitask` |
| `src/ctr_cvr/train.py` | 数据指纹、训练、early stopping、最佳权重、结果落盘 | `set_seed`、`sha256_file`、`main` |
| `src/ctr_cvr/evaluate.py` | 从 manifest 恢复处理器与模型，重载检查点独立复算 | `main` |
| `src/ctr_cvr/summarize.py` | 检查多 seed 可比条件并算均值/样本标准差 | `main` |
| `tests/` | 标签、切分、未来数据隔离、序列化和模型容量不变量 | 四个 `test_*.py` |
| `scripts/` | 安装、许可确认、下载、准备、主实验、消融、重载验证 | 按 README 顺序运行 |

## 关键函数逐层理解

### 1. `prepare_criteo.main`

它用 chunks 读取 653 MB 压缩源文件，使用固定 RNG 对每一行做 Bernoulli 采样，不查看 click 或 conversion。因此样本仍近似保留原始基准率。它记录源文件/准备文件 SHA256、源/样本总数和正例数，使“用的到底是哪份数据”可追踪。

禁止把“保留全部转化、下采样负例”的数据直接拿来报告原始 CTR、CVR 或未校正 LogLoss，因为采样已经改变概率分布。

### 2. `validate_and_label`

它先检查必需字段、二值标签和时间，再构造：

```text
ctr   = click
ctcvr = click AND conversion
```

随后稳定按 timestamp 排序。交集写法保证 `ctcvr <= click`，是 ESMM 概率链成立的标签前提。当前数据恰好没有 view-through conversion，但不能把这个偶然事实当接口合同。

### 3. `temporal_split`

目标是用过去预测未来。它寻找 70% 和 85% 附近的时间边界，并确保相同 timestamp 的行不会落入两侧。训练、验证、测试严格满足时间有序；测试集只做最终一次评价。

### 4. `FeaturePreprocessor.fit/transform/from_dict`

`fit` 只能访问训练段：保存连续特征均值/标准差和时间原点。`transform` 对验证、测试复用这些统计。离散 ID 使用固定哈希桶，限制 embedding 内存并自然处理未来未见 ID，代价是哈希碰撞。

完整特征除匿名 ID 外，还包含 `log1p(cost)`、小时 sin/cos 和距训练起点的天数。`from_dict` 让重载评估使用训练时完全相同的处理器，而不是重新拟合。

### 5. `build_model`

这个函数没有重写论文模型，而是配置 DeepCTR-Torch：

- DeepFM：线性/FM 学低阶交互，DNN 学高阶组合，只输出 CTR。
- ESMM：CTR tower 与 CVR tower；公开监督输出是 CTR 和两者乘积 CTCVR。
- MMoE：多个 expert、每任务 gate 和任务 tower；两个公开任务固定为 CTR/CTCVR。

`small/base/large` 改隐藏层容量，`num_experts` 只影响 MMoE。测试会验证 expert 数增加时参数量确实增加。

### 6. `evaluate_multitask`

全部曝光上评估 CTR 和 CTCVR；只对 `click==1` 行计算：

```text
pCVR = clip(pCTCVR / max(pCTR, epsilon), epsilon, 1-epsilon)
```

这与 `P(conversion|click)=P(click∩conversion)/P(click)` 一致。函数同时报告 `pCTCVR > pCTR` 的比例。ESMM 因乘法结构必为 0，MMoE 不受结构约束。

### 7. `train.main`

先固定 Python、NumPy、PyTorch seed 和线程数；真实数据先核验 SHA256。预处理只在训练段 `fit`。每个模型用验证集 BCE 做 Early Stopping 与 ModelCheckpoint，训练结束后明确从最佳 `.pt` 重载，再预测测试集。结果写入 data/environment/model/run 四类 JSON，并附检查点哈希。

本实验最大 10 epochs、patience 2；三模型在 3 个 seed 上都 best epoch=1，epoch 3 停止。这与“只设置 1 epoch”不同：后续 epoch 实际执行了，但验证目标未改善。

### 8. `evaluate.main`

它不是读取旧数字，而是：核对数据指纹和时间边界，从 `data_manifest.json` 恢复处理器，按 `run_manifest.json` 重建同一模型，加载最佳权重，再跑测试指标。seed 2026 三个模型的最大差异均为 0。

### 9. `summarize.main`

多 seed 求均值前，先要求同一准备数据 SHA256 和同一时间边界，并记录特征集、容量、最大 epoch 是否一致。标准差用 `ddof=1`，即样本标准差。它避免把不同数据或配置的运行偷偷混成误差条。

## 读结果时的四条纪律

1. DeepFM 只能与另外两者的 CTR 比；没有 CTCVR 输出就是不可比较，不是缺失值可以随意填补。
2. AUC 衡量排序，LogLoss 衡量概率质量；广告出价依赖概率值，不能只报 AUC。
3. clicked-CVR 只在点击人群上计算，样本更少、方差通常更大。
4. 三 seed 的小差异只能说“本实验观察到”，不能自动上升为统计显著或业务收益。

## 30 个项目面试追问与参考答案

### 数据与标签

1. **为什么不用 Avazu？** Avazu 公开监督标签只有点击，没有真实转化。人为制造转化标签不能称为 CVR 实验。
2. **一行数据代表什么？** 一次广告曝光以及曝光时可用的匿名上下文，后面附有点击和转化结果。
3. **CTR 标签是什么？** `click`，在全部曝光上训练与评估。
4. **CTCVR 为什么是交集？** 它表示曝光后既点击又转化的联合事件，必须满足 `CTCVR⊆CTR`。
5. **CVR 在哪里评估？** 只在 `click==1` 的曝光上，用 CTCVR 概率除以 CTR 概率得到条件概率。
6. **原始 conversion 能直接作为 ESMM 第二个标签吗？** 不能默认可以；如果含浏览后转化，就不再是 click-and-conversion 联合事件。当前文件虽无此类行，代码仍固定取交集。
7. **为什么均匀采样？** 它不利用标签选择样本，较好保留原始基准率，使 LogLoss 和预测均值仍可解释。
8. **10% 样本有多少数据？** 1,646,782 条；训练/验证/测试为 1,152,748/247,016/247,018。
9. **为什么没有跑全量？** 当前 pandas + DeepCTR-Torch 输入为内存式，实验机器 16 GB RAM、当时可用约 2.5 GB；全量有 OOM 风险。优先完成 10% 的多 seed、全模型、消融和重载闭环。

### 切分与泄露

10. **为什么不用随机切分？** 线上用过去预测未来；随机切分混入未来分布，会高估泛化并弱化分布漂移。
11. **为什么相同时间戳不能跨集合？** 同时发生的强相关曝光被拆开可能形成边界泄露，也会让“训练早于测试”不严格。
12. **只做时间切分就无泄露了吗？** 不是。还要限制特征可用时点、预处理拟合范围、观察窗口与实体重叠。
13. **哪些字段被禁用？** 转化时间/ID、归因、点击位置/次数、CPO、距上次点击时间；它们直接或间接来自事后行为。
14. **标准化怎样避免泄露？** 均值、标准差和时间原点只在训练段计算，验证/测试复用冻结值。
15. **时间字段本身会泄露吗？** 相对时间可作为曝光时可用特征，但可能学到漂移；不能用未来窗口统计。周期小时特征在曝光时已知。

### 特征与工程

16. **为什么用哈希桶而不是 LabelEncoder？** 高基数匿名 ID 需要可控 embedding 大小，也要处理未来未见值；哈希做到固定内存，代价是碰撞。
17. **为什么对 cost 做 `log1p`？** 降低长尾尺度和极端值影响，同时能处理 0。
18. **为什么小时用 sin/cos？** 小时是周期变量；23 点和 0 点应相近，单个整数无法表达环形邻接。
19. **数据 SHA256 有什么用？** 防止同名文件被替换或损坏，让训练、汇总和重载评估指向确切字节内容。
20. **固定 seed 就完全可复现吗？** 不一定。硬件、库版本、多线程和非确定性算子仍会影响；所以同时记录环境并报告多 seed。

### 模型

21. **DeepFM 为什么是强基线？** 它同时建模一阶、二阶和高阶特征交互，适合稀疏广告特征，不是简单 LR 陪跑。
22. **ESMM 解决什么问题？** 直接只在点击样本训练 CVR 会有样本选择偏差且正例稀疏；ESMM 在全曝光空间联合学习 CTR 与 CTCVR。
23. **ESMM 两个损失是什么？** 全曝光上的 click BCE 和 click-and-conversion BCE；不是 clicked-only CVR loss。
24. **ESMM 为什么保证 `pCTCVR≤pCTR`？** 第二个输出是两个 [0,1] 概率相乘，乘积不会超过 CTR 因子。
25. **MMoE 的 expert 和 gate 各做什么？** experts 学候选共享表示；每个任务的 gate 为 experts 分配不同权重，任务 tower 再输出概率。
26. **为什么 MMoE 也训练 CTR/CTCVR？** 固定监督标签与样本空间，避免比较模型时同时改变任务定义。
27. **expert 越多越好吗？** 本实验不是。5 experts 的 CTCVR 更好，但 CTR、clicked-CVR 和对应校准更差；3 experts 更均衡。

### 评估与实验

28. **AUC 提升而 LogLoss 变差怎么解释？** 排序更好但概率校准更差。若概率用于出价，应进一步看校准或做校准方法，不能只取 AUC。
29. **DeepFM 与 ESMM/MMoE 能公平比较什么？** 只能公平比较 CTR AUC/LogLoss；DeepFM 没有 CTCVR/CVR 输出，其他指标不可比较。
30. **你真正完成了什么，哪些不是原创？** 完成了数据源与许可核对、标签合同、时间切分、防泄露、统一训练/评价、三 seed、两类消融、记录、检查点重载、测试和文档。DeepFM/ESMM/MMoE 主体来自 DeepCTR-Torch；工程有 AI 辅助，不能声称从零手写或独立提出。

## 建议集中学习顺序

1. 手算三条样本的 click、conversion、ctcvr，说明 CVR 评价人群。
2. 阅读 `data.py` 和对应测试，画出时间切分边界与 fit/transform 权限。
3. 只跑 DeepFM，追踪输入字典、BCE、AUC/LogLoss 和最佳检查点。
4. 读 ESMM 输出关系，验证每条样本 `pCTCVR<=pCTR`。
5. 读 MMoE expert/gate/tower，解释为什么独立输出不保证包含关系。
6. 对照主实验 JSON 与表格，亲自复算一个均值和样本标准差。
7. 运行一次 `evaluate`，确认它真的从磁盘加载权重而不是复用内存模型。
8. 不看答案口述上面 30 问；答不稳的部分再回源码定位。
