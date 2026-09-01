# 简历项目描述与技术栈

## 推荐项目名称

**基于SecretFlow的横向联邦多阶段电影推荐系统**

英文：**Privacy-Preserving Multi-Stage Recommender with Federated DeepFM**

## 简历版项目描述

设计并实现“本地多路召回—横向联邦DeepFM精排—validation冻结融合”的完整电影推荐链路。基于MovieLens 100K将用户数据模拟拆分至两个参与方，构建热门度、ItemCF、内容偏好、带负反馈ItemCF和UserCF五路召回；通过SecretFlow FedAvgW与安全聚合协同训练DeepFM，并利用召回困难负样本、混合BCE/Focal Loss和用户内秩融合优化Top-10头部命中。在270名高暖意用户、500候选母池自然召回100的受控离线协议下，实现HitRate@10 91.11%、AUC 0.7535，并完成419用户扩展压力实验。

## 简历要点（推荐放3条）

- 搭建端到端多阶段推荐系统，使用热门度、ItemCF、内容偏好、Signed ItemCF与UserCF进行参与方本地召回，将500候选压缩至100后交由联邦DeepFM完成Top-10排序。
- 基于SecretFlow实现双参与方横向联邦训练，采用FedAvgW与SecureAggregator聚合模型更新；通过去除原始用户ID嵌入、50/30/10/10困难负采样和85%BCE+15%Focal提升稀疏候选排序鲁棒性。
- 建立validation选型、test冻结和标签泄漏审计机制；主实验在270名用户上达到HitRate@10 91.11%、NDCG@10 29.77%、AUC 75.35%，并通过419用户压力实验定位AUC与Top-10命中的目标错位问题。

## 技术栈

| 类别 | 技术 |
|---|---|
| 语言与数据 | Python、Pandas、NumPy |
| 推荐算法 | 多路召回、ItemCF、UserCF、内容推荐、DeepFM、秩融合、困难负采样 |
| 深度学习 | TensorFlow、Keras、Embedding、FM二阶交互、MLP、BCE/Focal Loss |
| 联邦学习 | SecretFlow、SecretFlow-FL、Horizontal FL、FedAvgW、SecureAggregator |
| 实验评估 | Precision@10、Recall@10、HitRate@10、NDCG@10、AUC、召回上界 |
| 工程工具 | JupyterLab、Git、GitHub、Pytest、JSON配置化实验 |

## 30秒口述

我做的是一个完整的联邦推荐项目，不是只训练一个排序模型。每个参与方先利用本地历史做五路召回，把500候选缩到100；然后两方在不汇总原始交互的前提下，通过SecretFlow进行横向FedAvgW训练DeepFM；最后只在validation上选择DeepFM和召回分数的用户内秩融合，冻结后评估test。在270用户受控实验中HitRate@10达到91.11%。扩充到419用户后AUC继续上升但HitRate略低于90%，我进一步定位到全局排序能力和Top-10头部命中目标并不完全一致。

## 两分钟STAR表达

**Situation：** 原始DeepFM直接面对大候选集时，HitRate@10随候选数快速下降；只有精排、没有召回的链路也不具备落地结构。

**Task：** 在横向联邦约束下补齐候选生成，并保证不利用test标签调参，逐步扩大用户规模与候选压力。

**Action：** 我先建立固定留出和标签审计；在参与方本地实现五路召回；针对召回候选增加困难负样本；使用DeepFM学习一阶、二阶和高阶交互；通过SecretFlow FedAvgW聚合两方模型更新；最后用validation选择小规模秩融合方案，冻结到test。同时输出召回上界、Recall、AUC和NDCG诊断召回与精排各自的损失。

**Result：** 132用户里程碑达到93.94%，扩展到270用户仍达到91.11%；继续扩到419用户后AUC达到0.7649、HitRate为89.98%，说明模型整体判别力提升，但Top-10边界仍有少量用户未命中，也明确了下一步应优化用户级排序目标而非继续盲目增加全局困难负样本。

## 表述边界

建议说“在受控500候选离线协议上达到91.11%”，不要说“线上全量推荐准确率91%”。

建议说“原始交互不跨参与方汇总，模型更新经安全聚合”，不要说“实现了绝对隐私”或“达到生产级密码安全”。SecretFlow文档本身也提示模拟SecureAggregator需要结合生产威胁模型进一步审查。

