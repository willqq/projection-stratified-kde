from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
E=R.parent/'enhancement_assessment_20260920'
b=pd.read_csv(E/'analysis/bandwidth_summary.csv')
b['classification']=np.where(b.matched_gain_percent<=0,'Reversed',np.where(b.matched_gain_ci_low>0,'Supported','Positive but uncertain'))
extras=[]
for (ds,phase),z in b.groupby(['dataset','phase']):
 v=pd.read_csv(E/'results/bandwidth'/f'{ds}_{phase}_variance.csv')
 n=720 if ds=='amazon' else 1800
 for h,q in v.groupby('h_factor'):
  cv2=q.uniform_relative_variance*128/(1-128/n)
  extras.append(dict(dataset=ds,phase=phase,h_factor=h,H_mass_median=q.H_mass_recall.median(),
   kernel_CV_median=float(np.sqrt(cv2).median()),single_relative_variance_median=q.single_relative_variance.median(),
   multi_relative_variance_median=q.multi_relative_variance.median(),variance_ratio_median_recomputed=q.variance_ratio.median()))
b=b.merge(pd.DataFrame(extras),on=['dataset','phase','h_factor'],validate='one_to_one')
assert np.max(np.abs(b.variance_ratio_median-b.variance_ratio_median_recomputed))<1e-12
b['exact_annular_opportunity_pp']=b.uniform_mc_error_percent-b.exact_annular_error_percent
b.to_csv(R/'analysis/bandwidth_classification.csv',index=False)
lines=['# Bandwidth mechanism reanalysis','',
'复用已冻结的全部35个条件，不新增抽样或选择带宽。20个test条件分为：Supported 15；Positive but uncertain 4；Reversed 1。15个同源audit条件单独统计，不能作为新增数据集。', '',
'| Set | Dataset | h/h0 | Classification | Gain % [95% CI] | Multi/Single variance | Median H mass | Kernel CV | Exact opportunity (pp) |',
'|---|---|---:|---|---|---:|---:|---:|---:|']
for z in b.itertuples():
 lines.append(f'| {z.phase} | {z.dataset} | {z.h_factor:g} | {z.classification} | {z.matched_gain_percent:.2f} [{z.matched_gain_ci_low:.2f}, {z.matched_gain_ci_high:.2f}] | {z.variance_ratio_median:.3f} | {z.H_mass_median:.3f} | {z.kernel_CV_median:.2f} | {z.exact_annular_opportunity_pp:.2f} |')
lines+=['',
'四个test的0.5h0条件全部未得到CI排除零的改善；其中GIST均值反向（CI仍跨零）。第五个未支持条件为GIST/0.75h0，均值正向但不确定。其余h0、1.5h0、2h0在四test与三audit上均有正下界。',
'',
'窄核失败不能归因为exact-stratification机会消失：0.5h0的MC−ExactAnnular差仍大。核值也未接近常数，CV在窄核更高。相同H后的Multi/Single方差比在0.5h0接近1，说明在该剩余总体里当前分层保留的独立方差收益较小。H吸收的质量随数据集变化，需按表解释，不能把ISOLET的现象推广到全部数据。',
'',
'宽核下kernel CV降低，同时分层的相对方差下降更明显；因此较高相对gain可以伴随较小绝对误差。本文应同时提供Single与Multi绝对误差。此处的解释是已测机制诊断，不能由几个聚合统计建立所有核/分布下的因果定律。',
'',
'分层估计器的正确性在各h保持；其优势幅度与可利用的余集结构有关。CI跨零表示证据不足，不能视为已证明无效；负均值也不等于显著反向。Recovery评估包含H的完整方法相对指定ExactAnnular的误差收益，不能替代matched-H独立增量。']
(R/'BANDWIDTH_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
# Preserve selected existing evidence tables, without recalculating or rewriting original data.
for name in ['scaling_recovery.csv','scaling_breakdown_mean_ms.csv','scaling_paired_comparisons.csv','deann_frontier_change.csv','deann_formal_summary.csv']:
 (R/'analysis'/('reused_'+name)).write_bytes((E/'analysis'/name).read_bytes())
paths=[E/'analysis'/n for n in ['bandwidth_summary.csv','scaling_recovery.csv','scaling_breakdown_mean_ms.csv','scaling_paired_comparisons.csv','deann_frontier_change.csv','deann_formal_summary.csv']]
paths+=sorted((E/'results/bandwidth').glob('*_variance.csv'))
(R/'qa/REUSED_INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print(b.groupby(['phase','classification']).size().to_string())
print(b[(b.phase=='test')&(b.h_factor==.5)][['dataset','H_mass_median','kernel_CV_median','variance_ratio_median','exact_annular_opportunity_pp']].to_string(index=False))
