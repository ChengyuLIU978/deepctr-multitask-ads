# 数据卡：Criteo Attribution Modeling for Bidding Dataset

## 身份与合法来源

- 发布方：Criteo（官方 Hugging Face 组织 `criteo`）。
- 数据页面：<https://huggingface.co/datasets/criteo/criteo-attribution-dataset>
- 官方研究数据入口：<https://ailab.criteo.com/ressources/>
- 许可：数据页面标注 **CC BY-NC-SA 4.0**。
- 原始压缩文件：约 653 MB；页面给出的 SHA256 为 `94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98`。

下载脚本要求显式传入 `-AcceptLicense`，只从上述官方组织下载并校验哈希。数据文件被 `.gitignore` 排除，本项目不重新分发数据。

这不是法律意见。若用途超出个人学习/非商业研究，必须重新阅读许可证并自行确认合规性。

## 为什么符合项目要求

每行是一次真实、匿名化的广告曝光，含：

- `timestamp`：曝光时间，可做时间切分；
- `click`：是否点击；
- `conversion`：是否发生转化；
- `uid`、`campaign`、`cat1..cat9`、`cost`：曝光时可用的匿名特征。

因此它同时提供曝光、点击和转化信号。Avazu 只有点击标签，不满足 CVR 项目的标签要求。

## 项目标签

- `CTR label = click`
- `CTCVR label = click AND conversion`
- `CVR label = CTCVR`，但只在点击样本上解释和评估

这样强制 `CTCVR ⊆ CTR`，符合 ESMM 的概率链：

`P(CTCVR | impression) = P(click | impression) × P(conversion | click, impression)`

原始 `conversion=1, click=0` 的非点击/浏览后转化不会当成后点击 CVR 正例。当前官方文件中，806,196 个转化均同时有点击，因此实际排除数为 0；代码仍显式取交集，并在每次运行的 `data_manifest.json` 中记录该数量，防止未来数据版本或其他数据源改变标签语义。

## 明确排除的泄露字段

| 字段 | 排除原因 |
|---|---|
| `conversion_timestamp` | 转化发生后才知道 |
| `conversion_id` | 直接暴露转化事件 |
| `attribution` | 归因结果是事后标签 |
| `click_pos`, `click_nb` | 由点击结果形成 |
| `time_since_last_click` | 与点击路径/结果强相关，定义时点不适合作为曝光时特征 |
| `cpo` | 依赖订单/转化结果 |

## 切分与采样

原始文件按时间读取。准备脚本执行不看标签的 Bernoulli 均匀采样；没有为了让结果好看而保留全部正例。正式实验使用 seed 2026 的 10% 样本：1,646,782 条曝光、594,818 次点击、80,789 次点击后转化；准备文件 SHA256 为 `0080fcbebdb6685d114077279fd45b8095310d019aabf92fdfbffa7fafedf1e1`。训练脚本再按曝光时间 70%/15%/15% 切分，相同时间戳不会跨集合。

所有特征处理统计只由训练段拟合。训练和独立重载评估会重新计算准备文件 SHA256，并与来源元数据核对；多 seed 汇总必须使用同一指纹与时间边界。

局限：

- 正式实验仍只覆盖全量数据的约 10%。当前内存式实现没有声称完成全量训练。
- 数据有观察窗口，靠近窗口末尾的曝光可能存在转化标签截尾。
- 匿名字段无法支持丰富业务解释。
- 固定哈希桶会发生碰撞；这是控制内存的工程折中。

项目已完成三个 seed 并报告样本标准差；若要做更强统计结论，仍应增加全量/流式训练和 bootstrap 置信区间。

