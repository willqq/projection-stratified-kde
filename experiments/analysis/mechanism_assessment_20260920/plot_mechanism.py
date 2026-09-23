from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];A=R/'analysis'
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})
ds=['isolet','cifar10','cifar10_gist512','amazon'];labels=['ISOLET','CIFAR','GIST','Amazon']
colors={'original_norm_angle':'#BD5A33','norm_angle_hamming_quantiles':'#BD5A33','projected':'#176B9A','random':'#898D94','exact_order':'#45865F'}
names={'original_norm_angle':'Original norm-angle','norm_angle_hamming_quantiles':'Hamming quantiles','projected':'Projected','random':'Random, five partitions','exact_order':'Exact-distance order'}
s=pd.read_csv(A/'fixed_union_summary.csv')
fig,axes=plt.subplots(1,3,figsize=(12.6,4.0),layout='constrained')
for ax,block,title,methods in zip(axes,['old_sizes','equal_C','matched_H'],['A  Fixed C: original cell sizes','B  Fixed C: equal-size cells','C  Full P: same exact subset H'],
 [['original_norm_angle','projected','random','exact_order'],['norm_angle_hamming_quantiles','projected','random','exact_order'],['projected','random','exact_order']]):
 for method in methods:
  z=s[(s.block==block)&(s.partition==method)].set_index('dataset').loc[ds]
  ax.plot(labels,z.predicted_variance_ratio_median,'o-',color=colors[method],label=names[method],lw=1.5,ms=4)
 ax.set_title(title,loc='left');ax.axhline(1,color='#CCCCCC',ls='--',lw=.8);ax.set_ylim(0,1.12)
 ax.set_ylabel('Median conditional variance / Single');ax.grid(axis='y',alpha=.18);ax.legend(loc='lower left',frameon=False)
for ext in ['png','pdf']:fig.savefig(A/f'fixed_union_mechanism.{ext}',dpi=180)
plt.close(fig)
z=pd.read_csv(A/'degradation_summary.csv')
fig,axes=plt.subplots(1,3,figsize=(12.6,3.7),layout='constrained')
cols=['#176B9A','#AF5C28','#45865F','#825598']
for d,label,color in zip(ds,labels,cols):
 zz=z[z.dataset==d]
 axes[0].plot(zz.level*100,zz.predicted_variance_ratio_median,'o-',label=label,color=color)
 axes[1].plot(zz.level*100,zz.mean_relative_error_percent,'o-',label=label,color=color)
 f=pd.read_csv(R/'results/fixed_union'/f'{d}_query_metrics.csv');f=f[(f.block=='matched_H')&(f.partition=='projected')]
 axes[2].scatter(f.predicted_relative_variance,f.empirical_relative_variance,s=10,alpha=.5,color=color,label=label)
axes[0].set_ylabel('Median conditional variance / Single');axes[1].set_ylabel('Mean relative error (%)')
for ax in axes[:2]:ax.set_xlabel('Positions randomly reordered (%)');ax.set_xticks([0,25,50,75,100]);ax.grid(alpha=.2)
axes[0].set_title('A  Variance increases with disorder',loc='left');axes[1].set_title('B  Error follows at aggregate level',loc='left')
axes[0].legend(frameon=False,fontsize=9);axes[2].set_xscale('log');axes[2].set_yscale('log')
axes[2].plot([1e-5,1],[1e-5,1],color='gray',ls='--',lw=1)
axes[2].set_xlabel('Predicted Var(estimate / target)');axes[2].set_ylabel('Empirical variance (1,024 draws)');axes[2].set_title('C  Independent sampling agrees',loc='left');axes[2].grid(alpha=.15)
for ext in ['png','pdf']:fig.savefig(A/f'ordering_degradation.{ext}',dpi=180)
plt.close(fig)
