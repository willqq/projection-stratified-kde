"""Read-only reuse of the existing validation controls; no new experiment."""
from pathlib import Path
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1]
B=R.parent/'sprint_final/results/group_diagnostics'
rows=[];queryrows=[]
for ds in ['isolet','cifar10','cifar10_gist512','amazon']:
    q=pd.read_csv(B/f'{ds}_query_diagnostics.csv').drop_duplicates('query_id')
    v=pd.read_csv(B/f'{ds}_variance.csv');v=v[(v.budget==128)&(v.J==8)]
    # The source duplicates the same random/ordered/old controls across score kinds.
    for qi in sorted(q.query_id):
        z=v[v.query_id==qi]
        assert z.candidate_kernel_mass_recall.max()-z.candidate_kernel_mass_recall.min()<1e-14
        assert z.candidate_size.nunique()==1
        qq=q[q.query_id==qi].iloc[0]
        controls={
            'old_annuli':z[(z.kind=='hamming')&(z.grouping=='old_annuli')].variance_ratio.iloc[0],
            'hamming_quantile':z[(z.kind=='hamming')&(z.grouping=='approx')].variance_ratio.iloc[0],
            'real_quantile':z[(z.kind=='real')&(z.grouping=='approx')].variance_ratio.iloc[0],
            'random_quantile':z[(z.kind=='hamming')&(z.grouping=='random')].variance_ratio.mean(),
            'ordered_quantile':z[(z.kind=='hamming')&(z.grouping=='ordered')].variance_ratio.iloc[0]}
        for key,val in controls.items():
            queryrows.append(dict(dataset=ds,query_id=qi,raw_row_id=int(qq.raw_row_id),grouping=key,
                candidate_size=qq.candidate_size,candidate_fraction=qq.candidate_ratio,
                mass_recall=qq.candidate_kernel_mass_recall,variance_ratio=val))
    subset=pd.DataFrame(queryrows);subset=subset[subset.dataset==ds]
    row=dict(dataset=ds,queries=len(q),mean_candidate_mass_recall=q.candidate_kernel_mass_recall.mean(),
        median_candidate_fraction=q.candidate_ratio.median(),median_largest_old_cell=q.old_largest_layer_fraction.median())
    row.update(subset.groupby('grouping').variance_ratio.median().to_dict());rows.append(row)
A=R/'analysis';A.mkdir(exist_ok=True)
pd.DataFrame(queryrows).to_csv(A/'existing_same_C_query_controls.csv',index=False)
s=pd.DataFrame(rows);s.to_csv(A/'existing_same_C_summary.csv',index=False)
txt=['# 既有机制证据审计','',
'范围为既有 80 validation queries / dataset、M128、H 为空。以下不新增采样或选择配置。随机分组先在每个 query 内平均五次，再取 query 中位数。', '',
'| Dataset | Mean mass recall | Old cells | Hamming quantiles | Real-score quantiles | Random | Ordered |',
'|---|---:|---:|---:|---:|---:|---:|']
for _,z in s.iterrows():txt.append(f'| {z.dataset} | {z.mean_candidate_mass_recall:.4f} | {z.old_annuli:.3f} | {z.hamming_quantile:.3f} | {z.real_quantile:.3f} | {z.random_quantile:.3f} | {z.ordered_quantile:.3f} |')
txt += ['',
'数值列为 conditional variance / 同一候选总体均匀抽样 variance。Hamming、Real、Random、Ordered 使用同样的八个分位层人数与分配；Old cells 的层人数不同，只作原构造诊断，不能把其差异完全归于排序分数。', '',
'逐 query 核对表明，同一 C 内五种分组的候选人数与 kernel-mass recall 完全相同，方差却不同。这支持 coverage 指标无法单独识别 partition quality。C=P 的既有 Table2A 提供另一组完整覆盖控制；它的 recall 恒为 1。', '',
'Amazon 需要保留：旧候选分组方差比约 0.381，明显低于 1，同时平均 mass recall 约 0.913。该例说明遗漏质量与层内方差是分别影响估计的因素，不能笼统说所有旧分组都没有收益。', '',
'未计算一个混合所有查询与机制的 Spearman 排名。固定 C 内 recall 恒定，相关系数无定义；跨 C 混合时，相对误差还含候选偏差与 query 难度。用解析 variance 对解析 MSE 再作相关分析也会重复分解公式。现有日志并未建立“recall 通常是弱预测器，而 variance 是跨机制更强预测器”的一般统计命题。', '',
'建议保留的 empirical insight：在本实验的固定总体与预算控制下，辅助排序应同时按其诱导的核值方差评价。该判断已有分层抽样原理和相关 KDE 文献基础；本文提供具体受控证据，不宜主张原则首创。', '',
'不建议为上述较宽命题新增一个配置搜索。它会引入 support、bias、预算与 H 的额外因素，成本超过本轮低风险增强范围。']
(R/'MECHANISM_EVIDENCE_AUDIT.md').write_text('\n'.join(txt)+'\n')
print(s.to_string(index=False))
