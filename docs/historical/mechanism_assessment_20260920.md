# Reproduce and verify

## 冻结输入

主稿：structural_revision_20260919/manuscript/ICASSP2027_EN_revised_final.pdf。哈希与全部21个稿件文件见BASELINE_FREEZE.json。
科学实现：59d86dbfaaeebb7bdaace151afcec047f3e1a360；原远程目录/root/autodl-tmp/icassp_sprint_20260916_0103。此次未修改该实现。现有未跟踪文件状态保存在qa/ENVIRONMENT.json。
新诊断目录：/root/autodl-tmp/icassp_mechanism_20260920。运行源码为scripts/mechanism.py，START.json同时记录源码SHA与预先协议SHA。原始数据、model、split与projection缓存继续使用冻结目录；各dataset的DONE.json含数据及模型哈希。

## 已执行的远程命令

环境：OMP_NUM_THREADS=8、OPENBLAS_NUM_THREADS=8、MKL_NUM_THREADS=8、NUMEXPR_NUM_THREADS=8、CUDA_VISIBLE_DEVICES为空；LD_LIBRARY_PATH指向原venv/lib。CPU及平台记录于qa/ENVIRONMENT.json。

```sh
/root/autodl-tmp/icassp_revision_20260915/.venv/bin/python scripts/mechanism.py fixed
# 下载并分析第一阶段；满足预定条件后保存ACTIVATION.md
/root/autodl-tmp/icassp_revision_20260915/.venv/bin/python scripts/mechanism.py degrade
```

工作目录为新诊断目录；代码会拒绝覆盖已存在stage。再次计算应使用新的隔离输出目录，并在导入mechanism模块后将模块OUT设置为新目录，复制同一EXPERIMENT_ROADMAP.md后调用fixed()；分析并满足条件才调用degrade()。保留原结果。无需重新训练或查看test。

## 本地分析

使用numpy、pandas、scipy与matplotlib。按顺序运行：

```sh
python scripts/analyze_mechanism.py fixed
python scripts/analyze_mechanism.py degradation
python scripts/reanalyze_existing.py
python scripts/build_reports.py
python scripts/plot_mechanism.py
python scripts/verify_mechanism.py
python scripts/verify_reused_evidence.py
```

现有证据重分析脚本从同级enhancement_assessment_20260920目录读取；该目录及原enhancement完整包保留。所有新机制分析可由本包results独立重算。冻结稿件哈希验证与旧证据完整验证需要原目录存在。

## 数据定义

*_query_metrics.csv：每query/partition的真实人数、分配、coverage、解析方差、实测方差、MARE和排序相关；random_repeat在group内保持独立标识。
*_strata.csv：每层人数、样本数、缩放核均值与样本方差、方差贡献。
*_estimates.npz：每个键保存1024个estimate/target；误差为abs(value−1)。C子集的target为f_C，fullP块为f。
*_cache.npz：原空间核值按同query最大logK缩放；原空间距离只作离线oracle；包含C/H/partition IDs和真实row IDs。f_C=target_scaled*exp(kernel_scale_log)，标准化估计可据此恢复。
退化*_partitions.npz：全部query/path/level的余集ID顺序。

模拟复用了核值缓存，适合检验条件分布；不将缓存重放时间报告为online query latency。正式延时仍来自已完成的独立计时。

## 统计

每partition1024次无放回抽样；random partition及degradation各5次。query为置信区间单位；1000次percentile bootstrap、seed20260920。原test/audit、projection、采样seed协议不变。没有跨实验取最好结果或配置选择。

## 验证覆盖

verify_mechanism从保存IDs和核值独立重算全部16,320行，核对每次估计的实测方差和MARE；检查完整互斥覆盖、预算、H、人数及分配。24个样例另与原生产估计器核对。远程与本地50文件哈希一致。verify_reused_evidence重新检查旧正式结果、validation冻结时间、DEANN选择、真实规模行清单和基线277文件哈希。
