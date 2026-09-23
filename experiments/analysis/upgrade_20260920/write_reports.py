from pathlib import Path
import pandas as pd,json
root=Path('/Users/qiuzhipeng/Downloads/ICASSP/upgrade_20260920');d=root/'results/formal_analysis'
A=pd.read_csv(d/'path_a_timing_bootstrap.csv');B=pd.read_csv(d/'ann_H_paired_bootstrap.csv');V=pd.read_csv(d/'ann_H_variance.csv');S=pd.read_csv(d/'summary.csv');P=pd.read_csv(d/'nondominance_all_points.csv');Q=pd.read_csv(d/'current_profile_mean_ms.csv')
labels={'isolet':'ISOLET','cifar10':'CIFAR-10-Small','cifar10_gist512':'GIST-512','amazon':'Amazon'}
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
aset=A[A.n==20000].sort_values('budget');at=[]
for r in aset.itertuples():
 z=S[(S.n==20000)&(S.budget==r.budget)].set_index('method')
 at.append([r.budget,f'{z.loc["projected_current","median_latency_ms"]:.3f}',f'{z.loc["projected_fast","median_latency_ms"]:.3f}',f'{100*r.relative_reduction:.1f}%',f'[{100*r.ci_low:.1f}, {100*r.ci_high:.1f}]%'])
a_table=mdtable(['M','原完整排序 / ms','等价分区 / ms','完整时延下降','配对 95% CI'],at)
bt=[];audit=[];cost=[]
for ds,label in labels.items():
 anchor='O' if ds=='amazon' else 'E'
 for phase,rows in [('test',bt),('audit',audit)]:
  z=B[(B.dataset==ds)&(B.phase==phase)&(B.budget==128)&(B.anchor==anchor)]
  if z.empty:continue
  r=z.iloc[0];vr=V[(V.dataset==ds)&(V.phase==phase)&(V.budget==128)&(V.anchor==anchor)].iloc[0]
  rows.append([label,f'{r.single_error*100:.2f}',f'{r.multi_error*100:.2f}',f'{r.relative_reduction*100:.1f}%',f'[{r.ci_low*100:.1f}, {r.ci_high*100:.1f}]%',f'{vr.median_variance_ratio:.3f}'])
 z=S[(S.dataset==ds)&(S.phase=='test')&(S.budget==128)].set_index('method')
 cost.append([label,f'{z.loc["projected_current","median_latency_ms"]:.3f}',f'{z.loc["ann_"+anchor+"_multi","median_latency_ms"]:.3f}',f'{z.loc["official_"+anchor,"median_latency_ms"]:.3f}'])
b_table=mdtable(['数据集','Single 误差 %','Multi 误差 %','相对下降','配对 95% CI','新条件方差比'],bt)
ba_table=mdtable(['Audit 查询集','Single 误差 %','Multi 误差 %','相对下降','配对 95% CI','新条件方差比'],audit)
c_table=mdtable(['数据集','原投影方法 / ms','ANN-H Multi / ms','官方 DEANN-E / ms'],cost)
prof=Q[Q.n==20000].iloc[0]
small=A[A.n<20000]
posit=int((B.ci_low>0).sum());points=P[P.method.str.startswith('ann_')&P.method.str.endswith('_multi')]
assert len(points)==70 and points.dominated_by_existing.all()
(root/'UPGRADE_FEASIBILITY.md').write_text(f'''# 升级可行性：两条路线均完成验证

结论：A 的完整查询收益在 GIST n=20,000 上成立。B 的统计增量可以迁移到强 ANN 子集，但当前实现的完整成本没有达到合并标准。

## 实际瓶颈

原报告的 78% 对应“排序加分层构造”。本轮在同一 GIST 20,000 参考点、M=128 的既有 100 个 audit 查询上分别测量：完整排序均值 {prof.original_sort_mean_ms:.3f} ms，ID 收集 {prof.original_id_gather_mean_ms:.3f} ms，分层切片 {prof.original_slicing_mean_ms:.3f} ms。三者合计 {prof.original_sorting_and_strata_mean_ms:.3f} ms；带诊断计时的总均值为 {prof.profile_total_mean_ms:.3f} ms。因此排序自身占 {100*prof.original_sort_mean_ms/prof.profile_total_mean_ms:.1f}%，合计占 {100*prof.original_sorting_and_strata_mean_ms/prof.profile_total_mean_ms:.1f}%。

这个构造步骤复制 n 个 ID，没有复制全部原空间向量。采样后的 M 个原空间向量复制另计；本例均值 {prof.selected_original_vector_copy_mean_ms:.3f} ms。投影均值 {prof.projection_mean_ms:.3f} ms，低维扫描 {prof.low_dim_score_mean_ms:.3f} ms。诊断子计时存在包含关系，不能重复相加；下面的速度结论采用独立的生产路径完整计时。

## A：保持统计规则的构造加速

多秩选择只确定 H 与八个层的边界。跨边界同分时退回原来的分数—ID 完整排序。H 成员、各层成员、层人数和分配规则保持一致，层内仍均匀无放回抽样。

780 个合成成员测试通过；216 个可能结果的完整分布枚举吻合；1,600 个既有 validation 查询—预算条件通过 ID 耦合验证。远程每个真实查询和预算也检查全部层的成员集合。原抽样函数与核函数未改动。

Validation 在 n=20,000 的三个预算上降时延 40.3%–43.0%，通过预先规定的 20% 准入线。正式结果如下：

{a_table}

原 n=720/1800 的 35 个正式设置只出现约 {100*small.relative_reduction.max():.1f}% 提速至 {abs(100*small.relative_reduction.min()):.1f}% 减速，没有同等收益。尚未测定精确的规模交叉点。

## B：强 ANN 子集上的统计空间

使用此前 validation 冻结的 DEANN E/T 检索配置。Single 和 Multi 固定相同实际 H，并使用相同总核预算。Multi 在补集上用八个投影分数层，Single 对同一补集均匀抽样；两者均覆盖整个 P。官方 DEANN 单独保留，其排列抽样及重叠修正并未改成我们的对照实现。

Validation 的 E 配置在四数据集均通过统计准入，正式 M=128 结果进一步确认 22.2%–33.5% 的误差下降。ANN-H 下的条件方差重新计算，未借用旧 Projected-H 结果。但所有 70 个原始规模 Multi 点都有更快且误差不高的已有方案，因此 B 不满足完整成本标准。

## 可追溯性

原 624 个文件已完整备份。远程原仓库基线为 2501679f43f01f3e233af16afb0c99e376cc49e9，正式实验代码冻结为 f872802ae4cca95ec69376d548c8389627de8761。原始代码与结果未覆盖。认证恢复后在独立 worktree 执行，凭据未进入交付文件。详见 UPGRADE_FREEZE.json、protocol/UPGRADE_PROTOCOL.md 和 REPRODUCE.md。
''')
(root/'UPGRADE_RESULTS.md').write_text(f'''# 最终实验结果

最终判断为 **A：可合并的实现增强**。主方法继续采用固定 64 维高斯投影形成等人数余集分层。ANN-H 混合方案保留为扩展实验，不替换主方法。

## 1. 等价分区的完整查询收益

以下使用既有 GIST n=20,000、100 个 audit 查询、五个抽样种子。每次查询随机交错方法顺序，计时取三次预热调用的中位数。配对区间按 query 重采样。

{a_table}

这是在相同成员集合和相同抽样分布下的完整时延下降。原 n=720/1800 的设置未通过加速线，变化为 0.3% 提速至 4.4% 减速。准确交叉规模未测。

分层内部排列会改变相同种子对应的样本 ID；A 不提供新的统计改进。尤其 n=20,000、M=256 的两次样本均值误差为 10.25% 与 9.94%，不能据此宣称更换分区算法降低了真实统计误差。构造与采样分布的等价性由成员检查和分布检查确定。

来源：results/formal_analysis/path_a_timing_bootstrap.csv 与 summary.csv。

## 2. 强 ANN-H 下的独立统计增量

下面统一为 M=128、误差优先的冻结 ANN 配置 E；Amazon 的 E 与原 O 相同。Single 是“相同 H 精确求值加单一均匀余集”的受控估计器，不命名为官方 DEANN。方差比是 Multi / Single 的逐查询条件方差比中位数。

{b_table}

对应 audit 结果：

{ba_table}

这些 audit 来自相同数据矩阵，已在此前研究中查看；它们不再是新的独立验证。Amazon 没有额外 audit 行。

全部 E/T、预算和 test/audit 的 {len(B)} 个配对条件中，{posit} 个点位的 95% 区间下限大于零。唯一未排除零的条件是 Amazon、T、M=512：降幅 11.8%，区间 [-1.1, 23.8]%。所有区间均未做多重比较校正。该负结果继续保留。

来源：ann_H_paired_bootstrap.csv、ann_H_variance.csv。每个 query 先平均五个抽样种子，再执行 1,000 次配对 query-bootstrap。

## 3. 统计增益尚未带来完整成本优势

M=128 的生产路径时延如下。这张表只提供成本量级；正式合并判断允许各方法使用不同 M。

{c_table}

在原 n=720/1800 的全部 70 个 ANN-H Multi 工作点中，每个点都被至少一个已有方案在误差和完整时延上同时支配。支配者包括官方 DEANN，也包括精确求和或原投影方法。它们均未进入本轮测得的完整曲线前沿。

对预先规定的 5%、10%、15%、20% 误差目标，以及 0.125、0.25、0.5、1、2、4 ms 时间上限，B 没有在两个相邻设置达到至少 10% 的完整成本收益。未达到的目标如实标记，无插值。完整记录见 fixed_error_latency.csv、fixed_time_error.csv、strongest_envelope_comparison.csv 与 nondominance_all_points.csv。

因此可以写“投影余集分层的统计增量可以迁移至强 ANN 选出的 H”；目前不能写“该混合方案完整查询优于强 DEANN”。这项失败属于当前实现和评测范围，未证明所有 ANN 与分层组合都无法取得计算收益。

## 4. 真实性与边界

四数据集的原始参考集、带宽、变换和查询行均保持不变。投影维度为 64，种子为 20260916，J=8，比例分配不变。正式原始规模评测覆盖五个预算、五个抽样种子、200 个查询及三次计时；GIST 大规模复核保持此前 100 查询、三预算的范围。没有追加调参或第三条路线。

真值只在离线评估代码中使用。实际核调用数与原空间距离次数分别保留；DEANN 的重叠修正没有计成免费操作。所有在线方法步骤都计时；共同的原始特征预处理与冻结主稿一样位于已变换查询计时之外。

本轮没有重新检验更窄带宽或更多投影种子。原稿中的窄带宽失败、规模对预算的依赖、Amazon Hamming 反例均保留。大规模 DEANN 比较仍限 M=64/128/256，不能推出所有预算上的最优性。

原始逐查询记录位于 raw_remote_results.tar.gz，包含完整 JSONL、计数、子计时、输入标识、来源哈希和各阶段 VERIFICATION.json。remote_source.tar.gz 与 remote_upgrade.patch 对应独立远程分支。结果汇总和图件脚本均随交付保存。
''')
(root/'MERGE_DECISION.md').write_text('''# 合并决定：A，可合并的实现增强

保留投影分层主方法，加入统计等价的多秩分区实现。GIST n=20,000 的完整查询时延下降 40.4%–42.9%，在三个连续预算上成立；M=128 从 1.801 ms 降为 1.036 ms。原 n=720/1800 没有相应收益，原完整排序实现和主表数值继续保留。

强 ANN-H 扩展实验没有达到方法增强的合并标准。它在固定 H、固定 M 下减少误差，证明余集分层的统计增量可以迁移；所有 70 个正式 Multi 点仍被已有方案支配。本轮停止这条路线，不继续加入新配置。

## 分别判断三个维度

| 判断 | 结论 | 证据含义 |
|---|---|---|
| 统计等价构造加速 | 成立，限已测 n=20,000 | 成员与抽样法不变，完整耗时降低 |
| 强 ANN-H 上的分层增量 | 成立 | M=128 test 误差下降 22.2%–33.5%，方差比 0.444–0.649 |
| ANN-H 混合方案完整成本优势 | 未成立 | 70 个 Multi 点均有已有方案同时更快且不更差 |

## 新增的研究价值

原研究关注廉价辅助排序能否保留距离分层的统计收益。此次证据补上了构造成本这一环：完整排序只为获得少量边界，却承担了约四分之三的查询时间；保持边界成员不变的选择实现显著减少了大参考集开销。

多秩选择是已有技术，本轮新增价值来自经过完整计时验证的实现与适用范围。它不改变论文的估计目标或理论创新边界。强 ANN-H 对照进一步支持统计机制的迁移，同时表明降低方差不足以自动获得端到端收益。

## 主稿的最小修改

摘要加入大规模下的时延下降范围。方法只解释等价边界选择与同分回退。计算实验补入正式完整时延与小规模回退边界。结论保留统计优势，并把新的计算证据限定在已测规模。

原 Figure 1、误差—时延主图、Tables 1/2 的实验数值保持不变，明确它们对应冻结的完整排序实现。强 ANN-H 扩展实验保留在配套报告和全部曲线中，不把失败的成本表现包装为新主方法。窄带宽与 Hamming 反例继续保留。

原稿及结果保存在 baseline_backup/，候选稿使用单独的 ICASSP2027_EN_revised_upgrade.tex / PDF，差异见 MANUSCRIPT.diff。代码保留在独立分支，未覆盖 main。作者信息仍按原稿留空，补齐信息和作者审阅后才能决定上传。

本判断依据可核验的新增事实，不作录用概率或审稿评分保证。
''')
print('Three final reports written')
