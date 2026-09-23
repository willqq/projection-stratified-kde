from pathlib import Path
import json
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'analysis'
names={'isolet':'ISOLET','cifar10':'CIFAR-10-Small','cifar10_gist512':'GIST-512','amazon':'Amazon'}
b=pd.read_csv(A/'bandwidth_summary.csv');s=pd.read_csv(A/'scaling_summary.csv');rec=pd.read_csv(A/'scaling_recovery.csv')
fair=pd.read_csv(A/'deann_formal_summary.csv');change=pd.read_csv(A/'deann_frontier_change.csv')
breakdown=pd.read_csv(A/'scaling_breakdown_mean_ms.csv');paired=pd.read_csv(A/'scaling_paired_comparisons.csv')
verification=json.loads((R/'qa/VERIFICATION.json').read_text());assert verification['status']=='PASS'
lost=change[change.frozen_paper_frontier&~change.expanded_frontier]
assert len(lost)==1 and lost.iloc[0].dataset=='amazon' and lost.iloc[0].budget==128

lines=['# EXPERIMENT RESULTS','',
'结论：本轮三项实验均完成。新增证据支持保留冻结的投影分层方法，并要求收窄一个原有计算结论。当前论文及原始结果未改动。', '',
'## 版本与执行完整性','',
'本机未找到字面名称 `ICASSP2027_EN_revised_final(1).pdf`。本轮使用上一轮最新交付 `structural_revision_20260919/manuscript/ICASSP2027_EN_revised_final.pdf`，SHA256 为 `6e5a69f26c5a32c439f4b4d981fb7b248739299bd092f8caa9b832f948276d37`；对应全文含三组 projection-seed 检查及 error–latency 图。作者尚未回复文件名确认，因此本文结论针对这份可核验版本。', '',
'全部新实验遵循预先写入并哈希冻结的 `EXPERIMENT_DECISION.md`。方法保持 Gaussian r64、seed20260916、J8、H=floor(M/4)、比例分配及原空间求核。源实现的统计计算没有修改。独立 worktree 的最终 commit 为 `2501679`；其前一版 `5e5b492` 执行 bandwidth/fairness，最终版只修正 scaling 的 nlist 记录并补充运行起始时间。两版脚本均已保存。', '',
'新增原始评测包含 bandwidth 140,000 条、DEANN 正式对照 206,400 条、scaling 正式对照 45,500 条估计记录；validation 搜索记录另存。原 bandwidth h0 的误差复现差异不超过 8.9e-16；scaling 的 n1800 与旧 audit 前100行误差完全一致。175 个远程文件的传输哈希通过，277 个冻结基线文件保持原样。', '',
'以下置信区间先在每个 query 内平均五个抽样 seeds，再进行 1,000 次 query bootstrap。区间未经多重比较调整。不同 bandwidth、budget 和 n 复用查询，不能把这些工作点当作独立数据集。', '',
'## E1：bandwidth 的机制外推','',
'固定全部四个 test 与三个同矩阵 audit query sets，每套200行；M128；仅将 h 调为 h0 的0.5、0.75、1、1.5、2倍。所有方法使用相同的新目标，R0保持原值。Single 和 Multi 的 H、总预算与原空间核完全相同。', '',
'下表为 test 的 matched-H 相对误差下降（%）。星号表示本轮 query-bootstrap 95% CI 下界大于0。', '',
'| Dataset | 0.5 h0 | 0.75 h0 | h0 | 1.5 h0 | 2 h0 |','|---|---:|---:|---:|---:|---:|']
for ds in names:
 z=b[(b.dataset==ds)&(b.phase=='test')].sort_values('h_factor')
 vals=[f'{x.matched_gain_percent:.2f}'+(' *' if x.matched_gain_ci_low>0 else '') for _,x in z.iterrows()]
 lines.append('| '+names[ds]+' | '+' | '.join(vals)+' |')
t=b[b.phase=='test'];au=b[b.phase=='audit']
lines += ['',f'Test：{int((t.matched_gain_ci_low>0).sum())}/{len(t)} 个组合的 CI 下界为正，增益中位数 {t.matched_gain_percent.median():.2f}%。Audit：{int((au.matched_gain_ci_low>0).sum())}/{len(au)}，增益中位数 {au.matched_gain_percent.median():.2f}%。在 h/h0=1、1.5、2 的全部21个 set×bandwidth 条件中，CI 下界均为正。', '',
'**主要边界在0.5h0。** 七个 query sets 的区间均含0。GIST test 的增益为 -8.73%（95% CI -27.24% 至7.40%），audit 为 -0.74%（-16.92% 至15.30%）。保留这些负点估计；它们不证明总体效应必为负，也不支持明确的正向增量。0.75h0 的 GIST test 同样未排除零。', '',
'同 H 下的 Multi/Single 条件方差比中位数，在四个 test sets 的0.5h0处为0.930–0.981，在2h0处为0.195–0.525。宽核下的分层收益较明确；窄核下相对于单一剩余层的方差改善很小。', '',
'Recovery 的解释需要配合 matched-H。0.5h0 下，ISOLET 的 H 核质量占比中位数为97.24%，Recovery 为1.931，但 matched-H gain 的区间仍含零。精确子集可以带来大幅整体改善，而多层剩余结构的额外增量较弱。Recovery 比较的是指定固定半径的 exact-annular 基线，不是最优分层下界。全部35个条件的 Recovery 分母 bootstrap 下界为正；Amazon 在2h0的 Recovery 为0.473，不能写成所有 bandwidth 均超过0.5。', '',
'没有发生真值浮点下溢、零估计或非有限相对误差。h0 的点估计与冻结结果一致；本轮 CI 使用预先固定的 bootstrap seed20260920，端点可能与旧稿所用种子略有差别。', '',
'科学结论：剩余分层的效果延续到多个 bandwidth，强度受核集中程度影响。可加入带边界的机制证据；不支持“任何 bandwidth 均获益”。', '',
'完整数据：`analysis/bandwidth_summary.csv`、`bandwidth_query_means.csv`；逐查询与解析方差记录在 `results/bandwidth/`；图为 `analysis/bandwidth_matched_H.png`。', '',
'## E2：reference-set scaling','',
'使用已有 GIST-512 原矩阵。Reference 数量为1000、1800、5000、10000、20000，形成嵌套集合；新增行不属于既有 train/reference/validation/test/audit。固定旧 audit 清单的前100行，所有 n 使用同样 queries。该 query count 与原稿200行不同，因此 n1800的汇总值也可不同；对同样前100行的逐查询复现差异为0。', '',
'预处理、h0、R0及方法配置固定。预算为64/128/256，五个抽样 seeds。DEANN在每个n上按同一预先规则只使用80个 validation queries选择配置。下面先看预先指定的 M128。', '',
'| n | Projected error (%) | Recovery [95% CI] | Projected median ms | Exact sum median ms | Sort+strata / total mean time |',
'|---:|---:|---|---:|---:|---:|']
for n in [1000,1800,5000,10000,20000]:
 f=s[(s.n==n)&(s.method=='final')&(s.budget==128)].iloc[0];ex=s[(s.n==n)&(s.method=='exact')].iloc[0]
 rr=rec[(rec.n==n)&(rec.budget==128)].iloc[0];br=breakdown[breakdown.n==n].iloc[0]
 lines.append(f'| {n} | {f.mean_error_percent:.3f} | {rr.recovery:.3f} [{rr.recovery_ci_low:.3f}, {rr.recovery_ci_high:.3f}] | {f.median_latency_ms:.3f} | {ex.median_latency_ms:.3f} | {100*br.sorting_and_strata_ns/br.time_ns:.1f}% |')
lines += ['',
f'15个 n×budget 工作点中，{int((rec.recovery>.5).sum())}个 Recovery 点估计超过0.5；其中 n≥5000 的9个点有{int(((rec.n>=5000)&(rec.recovery>.5)).sum())}个。该比例不足以说明改善不随规模变化。M128在较大n下的 Recovery 明显下降；n5000/M64 的95% CI还包含负值。', '',
'计算方面，M128 的完整时延从0.312 ms增至1.651 ms，n20000低于精确求和的4.800 ms，但两者误差不同，不能称为固定精度下的2.9倍加速。所有 M128 点均被本次预算范围内的某个对照支配。', '',
'较大的 M256 保留了正向端点：n5000、10000、20000的 Projected strata 在本次 **64/128/256** 网格内非支配。该结论包含所有已冻结DEANN配置与精确求和，但未检验DEANN的M512，不能外推为更广的连续工作区间。', '',
'| n, M256 | Projected error / ms | Validation-error DEANN error / ms | Paired error reduction vs that DEANN [95% CI] |',
'|---|---|---|---|']
for n in [5000,10000,20000]:
 f=s[(s.n==n)&(s.method=='final')&(s.budget==256)].iloc[0];de=s[(s.n==n)&(s.method=='deann_mean_relative_error')&(s.budget==256)].iloc[0]
 pp=paired[(paired.n==n)&(paired.budget==256)&(paired.comparator=='deann_mean_relative_error')].iloc[0]
 lines.append(f'| {n} | {f.mean_error_percent:.3f}% / {f.median_latency_ms:.3f} | {de.mean_error_percent:.3f}% / {de.median_latency_ms:.3f} | {pp.final_gain_percent:.2f}% [{pp.gain_ci_low:.2f}, {pp.gain_ci_high:.2f}] |')
lines += ['',
'这些点呈现精度与时延的取舍。n5000的误差差异区间含零；n10000与20000的同预算误差降低较明确，同时查询仍比该DEANN配置慢。n20000/M256的 Recovery 为0.729，区间为0.290–0.883，不能把点估计当作超过0.5的稳健保证。', '',
'Profile 显示：n20000/M128 的 query projection、低维scan、sorting+strata、original-space kernels、other 平均耗时分别为0.012、0.082、1.306、0.084、0.198 ms，合计1.682 ms。排序与建层合并项占77.7%。这里用均值保证各部分可加；主要性能表使用median，两者不得混加。', '',
'科学结论：统计收益仍可在较大reference set上出现，但明显依赖budget。完整代价受排序与建层增长限制；目前只支持一条GIST嵌套reference序列及固定h、固定投影种子。', '',
'完整数据：`analysis/scaling_summary.csv`、`scaling_recovery.csv`、`scaling_paired_comparisons.csv`、`scaling_breakdown_mean_ms.csv`；原始行在 `results/scaling/`；原始row-ID清单为 `results/scaling_ids.json`。', '',
'## E3：更充分的 DEANN 比较','',
'每个dataset检验24对 nprobe/near-fraction 参数，全部80个validation queries，M64/128/256，两个sampling seeds。先冻结最低误差、最低时延、误差×时延乘积的配置，去重后在原四个dataset各得到两条新选择曲线。原配置同时保留。正式评测包括全部200个旧test与audit queries、五个原预算和五个sampling seeds，三次 warmed calls；首个调用用于误差，三次中位数用于时延。各配置完整曲线均保存，没有在test上逐点挑选参数。', '',
'| Dataset | Validation-error choice (nprobe, fraction) | Validation-latency choice |',
'|---|---|---|']
fr=json.loads((R/'results/DEANN_FROZEN.json').read_text())
for ds in names:
 z=fr['datasets'][ds]['anchors'];lines.append(f"| {names[ds]} | ({z[0]['nprobe']}, {z[0]['near_fraction']}) | ({z[1]['nprobe']}, {z[1]['near_fraction']}) |")
lines += ['',
'| Dataset, M128 | Projected error / ms | Original DEANN error / ms | Validation-error DEANN error / ms |',
'|---|---|---|---|']
for ds in names:
 z=fair[(fair.dataset==ds)&(fair.phase=='test')&(fair.budget==128)]
 f=z[z.method=='final'].iloc[0];old=z[z.method=='deann_original'].iloc[0]
 choice=z[z.method=='deann_mean_relative_error'];de=choice.iloc[0] if len(choice) else old
 lines.append(f'| {names[ds]} | {f.mean_error_percent:.3f}% / {f.median_latency_ms:.3f} | {old.mean_error_percent:.3f}% / {old.median_latency_ms:.3f} | {de.mean_error_percent:.3f}% / {de.median_latency_ms:.3f} |')
lines += ['',
'旧DEANN和原方法误差均复现到浮点精度。新增validation-error配置在M128降低了ISOLET、CIFAR和GIST的平均相对误差，同时增加了完整时延；各预算的结果均保留。Amazon最低误差配置仍为原值，但新增低时延曲线改变了frontier。', '',
'| Query set | Frozen manuscript frontier | Same-run original configurations | Expanded DEANN comparison |',
'|---|---|---|---|']
for (ds,ph),z in change.groupby(['dataset','phase']):
 def budgets(k):return ', '.join(str(int(x)) for x in z[z[k]].budget) or 'none'
 lines.append(f'| {names[ds]} / {ph} | {budgets("frozen_paper_frontier")} | {budgets("rerun_original_configuration_frontier")} | {budgets("expanded_frontier")} |')
lines += ['',
'**需要修正的原结论：Amazon/M128。** Projected strata 为6.763% / 1.506 ms；新增DEANN低时延配置在M256为4.303% / 1.411 ms，平均实际kernel evaluations为276.89。虽然名义预算不同，这一完整查询工作点同时误差更低、时延更短。仅看同名M128会遗漏该支配关系。其余已写出的frontier点保留：Amazon/M32、M64和CIFAR/M512（test及audit）。', '',
'同轮只保留原配置时，所有原frontier判断均复现。因此该变化可以归于扩大的baseline覆盖，不能用跨运行时延漂移解释。Frontier仍为已测量均值/中位数的描述性判断，不是对新硬件、未测预算或所有IVF设置的保证。Amazon/M64与新DEANN同预算的时延差较小，尤其应保持这种限定。', '',
'本轮保留 nlist=min(32,floor(n/40))；扩大的是nprobe和near-fraction，并未穷举所有索引设计。DEANN的近邻重算、IVF全维距离、采样重叠修正和实际核调用均计入原始记录。当前实现的计时从同一个已变换query开始；原始预处理不包含在此完整估计器时间内，沿用冻结边界。', '',
'完整数据：`analysis/deann_formal_summary.csv`、`deann_frontier_change.csv`、`deann_formal_paired_comparisons.csv`；选择记录为 `results/DEANN_FROZEN.json`；所有候选的validation原始记录在 `results/deann_validation/`。', '',
'## 对一般 empirical insight 的判断','',
'既有同C控制已经显示：保持candidate population及mass recall相同，Hamming/real/random/ordered分组仍可得到明显不同的条件方差。新bandwidth检查又表明，即使final的full-population recall始终为1，matched-H增量仍随核集中程度变化。两者支持关注辅助排序所诱导的核值方差。', '',
'本轮未建立跨机制的“recall是弱预测器、variance总是更强预测器”统计排序。常量recall无法在固定C内计算Spearman；跨C混合又含遗漏偏差，解析variance与MSE的关系也有公式基础。应保留有限的、受控的经验结论，并继续承认已知投影和分层抽样组件的来源。详见 `MECHANISM_EVIDENCE_AUDIT.md`。', '',
'## 交付判断','',
'三项实验都有实质信息增益：bandwidth验证了有条件的机制延续；scaling量化了budget与排序成本的边界；DEANN压力测试发现一个原frontier点不再成立。最终决策为 **C. CLAIM NARROWING**。主方法、核心matched-H贡献和已有实验事实保留。主稿未编辑，具体最小更新建议在 `PAPER_UPDATE_DECISION.md`。']
(R/'EXPERIMENT_RESULTS.md').write_text('\n'.join(lines)+'\n')

decision='''# PAPER UPDATE DECISION

**C. CLAIM NARROWING**

保留当前方法及 matched-H 统计主线。新增实验提供了可用证据，同时要求收窄计算工作区间；当前冻结主稿保持原样。本判断针对 `structural_revision_20260919` 的可核验最新交付，字面名称带 `(1)` 的副本尚未得到作者确认。

## 为什么选 C

更充分的 validation-only DEANN 调优使 Amazon/M128 被 DEANN/M256 支配。原文“Amazon 的32、64、128三个预算处于frontier”以及 Conclusion 的“three Amazon budgets”需要改为两个。相同运行中的原配置重现了全部旧frontier，因此新增基线覆盖确实改变了结论。

Bandwidth 的正向结果值得补入：20个test组合中15个、15个audit组合中12个的matched-H增益CI下界为正；h0及更宽带宽的21个条件均为正。0.5h0处所有区间含零，这一边界必须同时说明。

GIST扩展到20k后仍可保留统计收益，但M128的Recovery仅0.321。M256在较大n有有限正向端点；这些点处于本轮预算范围的上端，未检验M512，更适合作为补充证据。排序与建层占n20k/M128平均时延约78%，计算边界不能仅用核求值数量描述。

## 最小必要更新范围（本轮仅提出，不编辑主稿）

1. **更新DEANN协议。** 在实验设置中说明24对参数、80 validation queries、三个validation budgets以及冻结整条配置曲线的选择规则。保存旧配置作为历史对照，主比较纳入新选最低误差和最低时延曲线。不能按test点选择配置后连成一条虚构的单配置曲线。

2. **修正具体frontier句子。** RQ4与Conclusion改为Amazon/M32、M64及CIFAR/M512（后者也出现在audit）。Amazon/M128退出，不能以同预算表格仍有取舍为由保留该claim。当前图表的比较区间要包含跨budget点与精确求和。

3. **同步M128的DEANN比较文字。** 若Table1采用validation-error配置，DEANN的error/median-ms应使用本轮对应值：ISOLET 8.256/0.143；CIFAR 10.427/1.226；GIST 10.311/0.207；Amazon 5.821/2.193。其余时延需使用同轮结果，原数值继续留在冻结包。原文“projected在CIFAR和GIST误差低于DEANN”应调整为针对具体配置的取舍描述；对新的validation-error配置，这句话已不成立。

4. **加入一句带宽结论及其边界。** 建议：`At M=128, the matched-subset benefit persists across multiple bandwidths, with positive lower bootstrap bounds in 15 of 20 test settings and 12 of 15 same-source audit settings; all intervals include zero at 0.5h0.` 具体倍率、固定H和统计规则放入精简补充表，避免让“多数条件”替代范围说明。

5. **替换larger n未测试的限制。** 可以改为：GIST已在固定100个query下扩至20k，M128 Recovery随该嵌套reference序列下降；排序成本增长使完整查询优势仍受限。保留M256的实际正向端点及区间，明确它们只在64/128/256网格中非支配。不要把与零误差exact sum的时延比写成固定精度speedup。

## 保持的内容

投影矩阵、r64、J8、H比例、比例分配、原空间核值和无偏条件均保持。Abstract与Introduction关于matched-H独立增量的主证据继续成立。三组projection seeds及原test/audit结果无需重跑。Norm-angle只承担诊断作用。

概念贡献继续使用“coverage alone does not determine partition quality”的准确边界。新结果支持一个具体的机制适用范围；它们没有建立新的分层抽样原理，也没有消除与Backurs/DEANN等工作的前史关系。

## 页面与信息取舍

优先修正已有计算比较，再用一两句话交代bandwidth与n边界。完整35条件bandwidth表、scaling profile和DEANN网格放入实验补充材料。保留Table2的matched-H核心证据；不为加入全部新图压缩字体或再开一个方法章节。

## 本轮完成状态

新增实验及原始记录：完成。

预先协议与validation冻结：通过核验。

冻结基线完整性：277个文件哈希一致。

正文修改：未执行。

下一版可维持“具有受控统计增量、计算适用范围有限”的定位，并应反映更强DEANN的比较结果。
'''
(R/'PAPER_UPDATE_DECISION.md').write_text(decision)
(R/'STATUS.md').write_text('''# Status

Completed: E1 bandwidth, E3 DEANN validation/freeze/formal, E2 GIST scaling.

Independent artifact verification: PASS. Baseline manuscript and 277 frozen files unchanged.

Decision: C. CLAIM NARROWING. Amazon/M128 leaves the observed frontier under expanded DEANN tuning. Matched-H statistical evidence remains supported within its measured bandwidth range.

No further parameter search or algorithm changes are planned in this task.

Version note: literal (1).pdf was not located; latest delivered structural revision was used provisionally, with exact PDF hash recorded. Author confirmation has not arrived.
''')
print('Reports written')
