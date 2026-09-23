from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
DS=['isolet','cifar10','cifar10_gist512','amazon']

def ci(v,stat=np.median):
 v=np.asarray(v);rng=np.random.default_rng(20260920)
 vals=stat(v[rng.integers(len(v),size=(1000,len(v)))],axis=1)
 return np.quantile(vals,[.025,.975])

def fixed():
 f=pd.concat([pd.read_csv(R/'results/fixed_union'/f'{d}_query_metrics.csv') for d in DS],ignore_index=True)
 numeric=['predicted_variance_ratio','empirical_relative_variance','predicted_relative_variance','mean_relative_error','ordering_spearman','relative_bias','empirical_over_predicted']
 # Random partitions are averaged within query before any cross-query inference.
 q=f.groupby(['dataset','block','partition','query_id'],as_index=False)[numeric].mean()
 rows=[];comparisons=[]
 for (ds,block,part),g in q.groupby(['dataset','block','partition']):
  single=q[(q.dataset==ds)&(q.block==block)&(q.partition=='single')].set_index('query_id')
  random=q[(q.dataset==ds)&(q.block==block)&(q.partition=='random')].set_index('query_id')
  g=g.set_index('query_id');uniform=single.loc[g.index]
  rows.append(dict(dataset=ds,block=block,partition=part,queries=len(g),predicted_variance_ratio_median=g.predicted_variance_ratio.median(),
   empirical_variance_ratio_median=(g.empirical_relative_variance/uniform.empirical_relative_variance).median(),
   empirical_over_predicted_median=(g.empirical_relative_variance/g.predicted_relative_variance).median(),
   empirical_over_predicted_p10=(g.empirical_relative_variance/g.predicted_relative_variance).quantile(.1),
   empirical_over_predicted_p90=(g.empirical_relative_variance/g.predicted_relative_variance).quantile(.9),
   mean_relative_error_percent=g.mean_relative_error.mean()*100,ordering_spearman_median=(g.ordering_spearman.median() if g.ordering_spearman.notna().any() else np.nan),
   absolute_relative_bias_mean=g.relative_bias.abs().mean(),
   mean_predicted_relative_variance=g.predicted_relative_variance.mean(),mean_empirical_relative_variance=g.empirical_relative_variance.mean()))
  if part=='projected':
   pv=random.loc[g.index].predicted_relative_variance;ev=random.loc[g.index].empirical_relative_variance
   reduction=100*(1-g.predicted_relative_variance/pv);empred=100*(1-g.empirical_relative_variance/ev)
   lo,hi=ci(reduction);elo,ehi=ci(empred)
   comparisons.append(dict(dataset=ds,block=block,predicted_reduction_median_percent=np.median(reduction),ci_low=lo,ci_high=hi,
     empirical_reduction_median_percent=np.median(empred),empirical_ci_low=elo,empirical_ci_high=ehi,
     empirical_over_predicted_median=np.median(g.empirical_relative_variance/g.predicted_relative_variance)))
 q.to_csv(R/'analysis/fixed_union_query_averages.csv',index=False)
 s=pd.DataFrame(rows);s.to_csv(R/'analysis/fixed_union_summary.csv',index=False)
 comp=pd.DataFrame(comparisons);comp.to_csv(R/'analysis/projected_vs_random_paired.csv',index=False)
 eligible=comp[comp.block.isin(['equal_C','matched_H'])]
 good=eligible[(eligible.predicted_reduction_median_percent>=10)&(eligible.ci_low>0)&(eligible.empirical_ci_low>0)&eligible.empirical_over_predicted_median.between(.8,1.2)]
 passed=good.dataset.nunique()>=2
 (R/'ACTIVATION.md').write_text('# Controlled degradation activation\n\n'+('PASS' if passed else 'SKIP')+'\n\n'+
 '按预先协议检查fixed-union结果。至少两个数据集在等人数C或固定H块中，投影相对随机分组的方差下降达到10%、配对query区间为正，且实测方差与解析方差没有系统不一致。\n\n'+('```csv\n'+comp.to_csv(index=False)+'```')+'\n\n'+
 ('满足启动条件，运行预定五级退化路径，不调整任何参数。' if passed else '未满足启动条件，不运行第二机制实验。')+'\n')
 print(s[s.partition.isin(['projected','original_norm_angle','norm_angle_hamming_quantiles'])].to_string(index=False))
 print('DEGRADATION_GATE',passed)

def degradation():
 f=pd.concat([pd.read_csv(R/'results/degradation'/f'{d}_query_metrics.csv') for d in DS],ignore_index=True)
 q=f.groupby(['dataset','query_id','level'],as_index=False).agg(predicted_variance_ratio=('predicted_variance_ratio','mean'),
  predicted_relative_variance=('predicted_relative_variance','mean'),empirical_relative_variance=('empirical_relative_variance','mean'),
  mean_relative_error=('mean_relative_error','mean'),ordering_spearman=('ordering_spearman','mean'))
 rows=[];changes=[]
 for (ds,level),g in q.groupby(['dataset','level']):
  rows.append(dict(dataset=ds,level=level,queries=len(g),predicted_variance_ratio_median=g.predicted_variance_ratio.median(),
   empirical_over_predicted_median=(g.empirical_relative_variance/g.predicted_relative_variance).median(),
   mean_relative_error_percent=100*g.mean_relative_error.mean(),ordering_spearman_median=(g.ordering_spearman.median() if g.ordering_spearman.notna().any() else np.nan)))
 for ds,g in q.groupby('dataset'):
  v=g.pivot(index='query_id',columns='level',values='predicted_variance_ratio')
  dif=v[1.]-v[0.];lo,hi=ci(dif)
  empirical=g.pivot(index='query_id',columns='level',values='empirical_relative_variance')
  error=g.pivot(index='query_id',columns='level',values='mean_relative_error')
  changes.append(dict(dataset=ds,endpoint_variance_ratio_increase_median=dif.median(),ci_low=lo,ci_high=hi,
    fraction_queries_nondecreasing_predicted=np.mean(np.all(np.diff(v.to_numpy(),axis=1)>=0,axis=1)),
    fraction_queries_nondecreasing_empirical=np.mean(np.all(np.diff(empirical.to_numpy(),axis=1)>=0,axis=1)),
    fraction_queries_nondecreasing_MARE=np.mean(np.all(np.diff(error.to_numpy(),axis=1)>=0,axis=1))))
 q.to_csv(R/'analysis/degradation_query_averages.csv',index=False)
 pd.DataFrame(rows).to_csv(R/'analysis/degradation_summary.csv',index=False)
 pd.DataFrame(changes).to_csv(R/'analysis/degradation_trends.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(changes).to_string(index=False))

if __name__=='__main__':
 import sys
 fixed() if sys.argv[1]=='fixed' else degradation()
