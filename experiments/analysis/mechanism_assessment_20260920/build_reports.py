from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];A=R/'analysis';E=R.parent/'enhancement_assessment_20260920'
ds=['isolet','cifar10','cifar10_gist512','amazon'];names=dict(zip(ds,['ISOLET','CIFAR-10-Small','GIST-512','Amazon']))
s=pd.read_csv(A/'fixed_union_summary.csv');q=pd.read_csv(A/'projected_vs_random_paired.csv');d=pd.read_csv(A/'degradation_summary.csv');tr=pd.read_csv(A/'degradation_trends.csv');b=pd.read_csv(A/'bandwidth_classification.csv')
def table(frame,cols,labels=None):
 labels=labels or cols
 lines=['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for _,z in frame.iterrows():
  lines.append('| '+' | '.join(f'{z[c]:.4f}' if isinstance(z[c],(float,np.floating)) else str(z[c]) for c in cols)+' |')
 return '\n'.join(lines)
rows=[]
for name in ds:
 p=s[(s.dataset==name)&(s.block=='matched_H')].set_index('partition')
 comp=q[(q.dataset==name)&(q.block=='matched_H')].iloc[0]
 rows.append(dict(dataset=names[name],pred=p.loc['projected','predicted_variance_ratio_median'],emp=p.loc['projected','empirical_variance_ratio_median'],random=p.loc['random','predicted_variance_ratio_median'],oracle=p.loc['exact_order','predicted_variance_ratio_median'],single_error=p.loc['single','mean_relative_error_percent'],multi_error=p.loc['projected','mean_relative_error_percent'],reduction=100*(1-p.loc['projected','mean_relative_error_percent']/p.loc['single','mean_relative_error_percent']),vs_random=f"{comp.predicted_reduction_median_percent:.1f}% [{comp.ci_low:.1f}, {comp.ci_high:.1f}]"))
matched=pd.DataFrame(rows);matched.to_csv(A/'matched_H_mechanism_table.csv',index=False)
texts=['# Experiment results — frozen-paper mechanism assessment','',
'本轮完成两个新机制实验，并重分析已有带宽数据。主稿、投影方法及正式test/audit结果保持不变。所有新增机制查询来自既有validation集，每数据集80个，不能称作新的独立测试。',
'',
'## 1. 执行与统计单位','',
'固定M=128、原h、原参考总体和预处理、r=64/seed20260916。norm-angle诊断复用L=12/c=5/b=2/J=8、24 epochs的原模型及floor分箱；它是主稿指定的诊断配置，不代表历史所有配置的最优结果。每partition进行1,024次独立无放回抽样，层间抽样独立。相同人数的partition之间复用随机位置以便配对。五个随机partition或退化路径先在query内平均，bootstrap只重采样query。所有预定数据集与结果保留；没有参数搜索或按结果追加抽样。',
'',
'保存8,320个fixed-union query–partition结果及8,000个degradation结果，共16,711,680个标准化估计值。核值来自原空间。为低成本验证条件分布，核值在离线缓存后重放；这些重复不代表新的完整在线时延测量。每次逻辑预算仍为128；物理核计算不重复计入16M次重放。',
'',
'## 2. 同总体下，分组是否改变方差？','',
'三个块回答不同问题：old_sizes固定旧candidate union及旧人数向量；equal_C在同一个candidate union上固定八个等人数层；matched_H覆盖完整P，精确H固定为投影选出的32个点，余集预算96。前两个块以f_C为目标；最后一块以完整f为目标。每个块内的coverage、H、层人数及分配均相同，Single仅用作同总体的均匀方差基准。',
'',
'### 最接近最终方法的matched-H结果','',
'以下为80个validation查询的条件方差比中位数；方差分母均为同H的Single。误差列是1,024次重放后跨query的平均相对误差百分数，不替换论文200-query正式结果。',
'',table(matched,['dataset','pred','emp','random','oracle','single_error','multi_error'],['Dataset','Projected predicted','Projected empirical','Random predicted','Ordered predicted','Single MARE %','Multi MARE %']),
'',
'四数据集projected的条件方差比约0.55–0.66，实测与解析一致。与同人数随机分组相比，其query级方差下降中位数为34.4%–44.9%，配对区间均为正。',
'',table(matched,['dataset','vs_random'],['Dataset','Variance reduction vs matched random [95% CI]']),
'',
'### 保留的反例与原构造诊断','',
'ISOLET保留旧层人数向量时，projected和原norm-angle的解析方差比中位数都约1.005，exact ordering也几乎无法降低方差。换成八个等人数层后，同C的projected比值为0.627。此处同时说明层人数设计的重要性；不能将跨块改善全归于投影排序。',
'',
'Amazon在固定C的等人数块中，Hamming quantiles为0.290，优于projected的0.629。原norm-angle在旧人数块也已有0.381的条件方差比。projection不是每个条件下最佳分组；旧方法的coverage和partition问题需分别评价。这个条件目标f_C的结果不能当作full-target算法胜负。',
'',
'ISOLET/old_sizes中，实测projected相对random出现约2.95%的中位方差下降，而解析下降约0，区间跨零。1,024次有限重放仍有Monte Carlo误差，尤其稀疏高核值查询。这个小差异不作为增益证据。对于其他清楚效应，解析与实测方向一致。所有逐query结果仍保留。',
'',
'## 3. 主动退化排序后的机制链','',
'第一实验满足预定启动条件后，执行exact余集排序的0/25/50/75/100%位置重排。H、总体、层人数、预算及h保持不变；每query固定五条路径。所有级别完整报告。',
'',table(d,['dataset','level','predicted_variance_ratio_median','empirical_over_predicted_median','mean_relative_error_percent','ordering_spearman_median'],['Dataset','Reordered fraction','Variance / Single','Empirical / predicted','MARE %','Ordering Spearman']),
'',
'四数据集的汇总方差及MARE均随退化增加，方差比从0.201–0.489上升到约1。对每query先平均五条路径后，解析方差在全部320个query上随级别非递减；实测方差的逐query非递减比例为73.8%–95.0%，MARE为80.0%–93.8%。实测不应被描述为逐query必然严格单调。',
'',table(tr,['dataset','endpoint_variance_ratio_increase_median','ci_low','ci_high','fraction_queries_nondecreasing_empirical','fraction_queries_nondecreasing_MARE']),
'',
'此处是对排序扰动的受控经验关系。不同误差分布可以具有相同方差而不同MARE；因此不宣称variance单独决定平均绝对相对误差。',
'',
'## 4. 带宽结果的完整分类','',
'四test的20个条件：Supported=15；Positive but uncertain=4；Reversed=1。三个同源audit的15个条件：12、2、1。未显著的test条件为四个0.5h0，加GIST/0.75h0；反向均值为GIST/0.5h0，CI仍跨零。全部h0、1.5h0、2h0条件支持matched-H正向增量。',
'',
'窄核下exact-annular opportunity没有消失；0.5h0的MC−ExactAnnular仍为19.2–63.1个百分点，kernel CV中位数为6.1–23.3，远非接近常数。与此同时Multi/Single方差比上升至0.930–0.981，显示独立分层收益减弱。ISOLET的H吸收核质量中位数达97.2%，其他数据集为45.7%–85.0%，不能用一个统一的H解释覆盖所有数据。具体35个条件见BANDWIDTH_ANALYSIS.md和analysis/bandwidth_classification.csv。',
'',
'## 5. 规模与DEANN：复用结果，分别判断','',
'GIST已测n=1k/1.8k/5k/10k/20k，固定同100个原audit查询。无需重复规模实验。15个n×M中11个Recovery点估计>0.5，6个CI下界>0.5；大n的结果依赖预算。20k时M128 Recovery=0.321，M256=0.729。后者CI较宽，不能称所有规模稳定达到同样收益。',
'',
'5k/10k/20k的M256在已测64/128/256网格中非支配。10k和20k相对validation-error DEANN的同预算误差下降约14.4%和11.0%，配对区间为正；其完整时延仍较长。未测大n/M512，故这些属于有限网格端点。20k/M128平均时延约78%来自sorting+strata；旧日志未分离纯sort与建层，不将该数字写成纯排序耗时。DEANN检索及原空间求和以现有完整时延/计数报告，不虚构未记录的分项。',
'',
'DEANN调优记录经过重新核验：24个参数对、全部80 validation queries、三个validation预算及两seed，依据聚合validation指标冻结整条配置曲线，然后评测test/audit。原配置保留并同期计时。该网格仍有范围限制，nlist固定，不能视作DEANN最优性能包络。',
'',
'新DEANN使Amazon/M128退出frontier；Amazon/M32、M64及CIFAR/M512（含CIFAR audit）仍保留。Amazon/M128被更大预算但更快、更准的DEANN/M256支配，因此完整时延比较应修正。跨预算计算支配不否定同H同M的统计分层增量。所有frontier均为测得点估计，未宣称时延差异有统计显著性。',
'',
'## 6. 能支持到哪个机制层级？','',
'Level1与Level2直接得到支持：完全相同的coverage可以对应明显不同的条件方差；固定人数和分配后，partition本身仍改变估计方差。',
'',
'Level3得到有边界的支持：在当前四数据集、固定h0/M128/H下，主动破坏余集排序使层内核值混合，诱导方差及汇总误差同步增加。可以据此建议按诱导方差评价辅助排序。这里未直接操纵nearest-neighbor recall，不能升级成“近邻质量不重要”或“variance是跨所有算法唯一预测量”。',
'',
'解释量应写为预算与有限总体修正加权的条件方差，而非未经加权的层内方差均值：V=(1/n²)Σ_j N_j²(1−m_j/N_j)S_j²/m_j。在full coverage且无偏条件下，它等于MSE；对应相对MSE为V/f²。层人数或分配改变时，权重也改变。以上原理属于已有分层抽样理论；本轮增加的是当前辅助构造、失败配置及排序干预的经验解释。',
'',
'## 7. 核验与交付','',
'qa/VERIFICATION.json从保存的原空间核值与membership独立重算16,320行方差和16,711,680个重放估计统计。并用24个确定性样例对照未修改的production estimator。qa/REUSED_EVIDENCE_VERIFICATION.json核验已有带宽、规模及DEANN流程。完整脚本、partition IDs、每次估计与两张机制图已保存。主稿目录21个文件哈希保持一致。']
(R/'EXPERIMENT_RESULTS.md').write_text('\n'.join(texts)+'\n')
(R/'PAPER_DECISION.md').write_text('''# Paper decision

**MECHANISM UPGRADE — 限于受控经验解释。**

新增结果值得增强当前论文的机制论证。方法保持冻结；计算适用范围同步局部收缩。主稿尚未修改。

## 对四个最终问题的回答

**为什么提升真实？** 固定H、人数与预算后的projected余层仍降低方差，1,024次抽样的实测方差与解析量吻合。已有200-query test/audit及多种投影seed的matched-H误差结果提供另外的支持。新validation机制实验不能充作新的独立测试。

**为什么超出单一设置？** 已有三个投影seed、三个预算、多个带宽及真实reference扩展说明存在可重复工作区间。窄核中的增量减弱，大n的Recovery依赖预算，结论须带这些范围。对M128的一次机制实验不能单独证明所有M和h。

**价值如何超过组件列表？** 受控结果揭示一个具体失败机制：ISOLET的旧人数向量几乎让任何排序失去作用；同C改为等人数层后，分组差异才明显。排序退化进一步连接了ordering、induced variance和实际误差。Amazon中Hamming优于projection的反例提醒读者，应评价所形成的核值层，而非预先认定某种表示优越。这是一项可核验的经验认识，仍不足以声称新抽样原理或首次投影分层。

**什么量决定价值？** 在总体、H、人数与分配固定且条件无偏时，预算加权并含有限总体修正的条件方差V直接决定MSE。它为辅助ordering提供任务相关评价。MARE还与误差分布有关；全局排序相关或coverage都不能代替这一条件量。

## 对录用证据和创新解释的判断

增强程度：机制解释有实质改善，实验覆盖也更清楚。它能直接回应“是否只是组合已知组件”和“怎样隔离分层增量”的质疑；这些证据无法量化录用概率。Backurs已有投影分区与无偏partition estimator，DEANN已有exact subset加uniform remainder，必须继续准确引用。

最高可用表述：Level1/2充分支持；Level3在预定排序干预和当前设置内支持。避免推广为任意kernel、任意auxiliary score、任意budget都成立的经验定律，也不宣称recall与误差普遍弱相关。

## 值得进入论文的最少内容

优先加入一段受控机制结果：同H和预算下，四数据集条件方差比中位数0.55–0.66，独立重复抽样验证其预测；排序退化使汇总方差和误差增大。Table2保留matched-H主证据，可用一个精简说明或小型补充表连接解析与实测方差，完整机制图留在实验材料。

保留一个有解释力的失败例：原ISOLET层人数失衡时，精确排序也几乎不降方差。这里须明确按原人数向量比较，避免把它误写成“精确分层通常无效”。Amazon的有利Hamming结果保留在补充材料及机制讨论，不能以投影优势的单向叙事覆盖。

带宽只需一句含边界的结果，完整35条件表保留在记录中。规模结果可替换larger n untested的限制，完整曲线和分项成本不挤占正文。

DEANN更新属于必要事实修正：Amazon的frontier预算由三个改为32/64两个；CIFAR512保留。相应图、表和描述应使用同轮公平比较，不保留已被新配置推翻的比较句。

## 主稿最多允许的局部修改范围

在作者决定采纳后，仅修改Section4.2的一段机制解释及Table2的必要说明；Section4.3补一句带宽范围；Section4.4更新DEANN协议、frontier与larger-n边界；Conclusion加一句机制结论并修正Amazon预算数量。若Abstract改动，只允许替换一句现有结果总结；Title、Method、估计公式和Introduction结构保持。

此处列出上限，不要求全部修改。优先用较弱的旧诊断文字腾出空间，维持4页技术正文。两张完整新机制图不同时进入主稿，不压字体。此次没有编辑或重新编译主稿。

## 留在实验记录中的内容

全部逐query/partition和1,024次重放；五条退化路径；原人数、等人数、matched-H三个对照块；包含反向结果的全部带宽条件；有限重放出现的小幅假象；完整DEANN调优与规模曲线；oracle控制和所有ID/计数。

## 下一步判断

当前证据足以支持有限的机制增强，继续增加模型或参数搜索的信息增益低。本轮实验停止。核心统计结论得到加强，计算结论按实测范围表述，主稿保持冻结供作者决定采纳哪些证据。
''')
(R/'README.md').write_text('''# ICASSP mechanism assessment

建议先读 PAPER_DECISION.md，再看 EXPERIMENT_RESULTS.md。RESEARCH_GAP_ANALYSIS.md与EXPERIMENT_ROADMAP.md为新实验前冻结的设计。

- analysis/fixed_union_mechanism.png：同总体下分组质量对照。
- analysis/ordering_degradation.png：排序退化、方差及误差的联系。
- BANDWIDTH_ANALYSIS.md：全部带宽条件的分类和解释。
- results/fixed_union：320个validation查询，三个受控块，每分组1,024个估计。
- results/degradation：五级排序退化，五条固定路径，完整估计和partition IDs。
- qa：源文件、协议、传输、公式、预算及已有证据复核。

新机制结果不替代冻结论文的正式test/audit结果。conditioned replay使用预存原空间核值，适合估计分布验证，不提供新的在线加速证据。最终论文文件未改。
''')
print(matched.to_string(index=False))
