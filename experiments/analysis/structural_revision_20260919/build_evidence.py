from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'evidence/analysis'; M=ROOT/'manuscript'
s=pd.read_csv(A/'summary.csv'); b=pd.read_csv(A/'paired_bootstrap.csv'); v=pd.read_csv(A/'fixed_c_variance.csv')
datasets=['isolet','cifar10','cifar10_gist512','amazon']
names=['ISOLET','CIFAR-10-Small','GIST-512','Amazon']
methods=['uniform_mc','exact_annular','final','deann']
labels=['Uniform MC','Exact annular','Projected strata','DEANN']
colors=['#555555','#276CA5','#C33F22','#478348']
markers={32:'o',64:'s',128:'^',256:'D',512:'v'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.4,'axes.labelsize':9.4,'axes.titlesize':9.7,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42,'ps.fonttype':42})

for phase,output in [('test',M/'fig3_latency'),('audit',ROOT/'supplement/error_latency_audit')]:
 fig,axes=plt.subplots(2,2,figsize=(7,3.2))
 for ax,d,name in zip(axes.flat,datasets,names):
  if phase=='audit' and d=='amazon':
   ax.axis('off');ax.text(.5,.5,'Amazon: no unused audit rows',ha='center',va='center');continue
  q=s[(s.phase==phase)&(s.dataset==d)]
  for meth,label,col in zip(methods,labels,colors):
   z=q[q.method==meth].sort_values('budget')
   ax.plot(z.median_latency_ms,z.mean_relative_error*100,color=col,lw=1,alpha=.85)
   for row in z.itertuples():ax.scatter(row.median_latency_ms,row.mean_relative_error*100,color=col,marker=markers[row.budget],s=23,zorder=3)
  z=q[q.method=='exact'].iloc[0]
  ax.scatter(z.median_latency_ms,0,color='black',marker='*',s=65,zorder=4)
  ax.set_title(name,pad=3);ax.set_xscale('log');ax.set_ylim(bottom=-2)
  ax.grid(alpha=.2,lw=.5);ax.spines[['top','right']].set_visible(False)
  from matplotlib.ticker import FuncFormatter,LogLocator,NullFormatter
  ax.xaxis.set_major_locator(LogLocator(base=10,subs=[1,2,5]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
 fig.text(.012,.43,'Mean relative error (%)',rotation=90,va='center',ha='left',fontsize=9.4)
 for ax in axes[1,:]:
  if ax.axison:ax.set_xlabel('Complete-query median (ms, log scale)')
 handles=[Line2D([],[],color=c,lw=1.5,label=l) for c,l in zip(colors,labels)]+[Line2D([],[],color='black',marker='*',ls='',label='Exact sum')]
 fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.51,1.015),frameon=False,columnspacing=.8,handlelength=1.3)
 fig.legend(handles=[Line2D([],[],color='#444444',marker=mk,ls='',label=f'M={budget}') for budget,mk in markers.items()],ncol=5,loc='upper center',bbox_to_anchor=(.51,.942),frameon=False,columnspacing=1.3,handlelength=.8)
 fig.subplots_adjust(left=.09,right=.985,top=.77,bottom=.13,wspace=.27,hspace=.65)
 fig.savefig(str(output)+'.pdf');fig.savefig(str(output)+'.png',dpi=220);plt.close(fig)

def get(df,phase,dataset,**kw):
 z=df[(df.phase==phase)&(df.dataset==dataset)&(df.budget==128)]
 for key,value in kw.items():z=z[z[key]==value]
 assert len(z)==1,(phase,dataset,kw,len(z))
 return z.iloc[0]

lines=[r'\begin{tabular}{lccccccc}',r'\toprule',r'& \multicolumn{3}{c}{A: fixed-population variance ratio} & \multicolumn{4}{c}{B: matched-$H$ remainder comparison} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-8}',r'Dataset / queries & Projected & Random & Ordered & Single (\%) & Multi (\%) & Gain (\%) & 95\% CI \\',r'\midrule']
ledger=[]
for phase in ['test','audit']:
 for d,name in zip(datasets,names):
  if phase=='audit' and d=='amazon':continue
  vari=[get(v,phase,d,grouping=g).median_variance_ratio for g in ['approx','random','ordered']]
  single=get(s,phase,d,method='matched_single').mean_relative_error*100
  multi=get(s,phase,d,method='final').mean_relative_error*100
  boot=get(b,phase,d,method='final',comparator='matched_single')
  gain=boot.relative_error_reduction*100;lo=boot.reduction_ci_low*100;hi=boot.reduction_ci_high*100
  label=name+(' / audit' if phase=='audit' else ' / test')
  lines.append(label+' & '+' & '.join(f'{x:.3f}' for x in vari)+f' & {single:.1f} & {multi:.1f} & {gain:.1f} & [{lo:.1f}, {hi:.1f}]'+r' \\')
  ledger.append(dict(phase=phase,dataset=d,Single=single,Multi=multi,gain=gain,ci_low=lo,ci_high=hi,variance=vari))
 if phase=='test':lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}']
(M/'table2_final.tex').write_text('\n'.join(lines)+'\n')
pd.DataFrame(ledger).to_csv(ROOT/'evidence/table2_values.csv',index=False)
(ROOT/'evidence/figure_source.json').write_text(json.dumps({'figure3':'All recorded test budgets; no interpolation or fitted frontier; lines only join measurements','source':'analysis/summary.csv','sha256':hashlib.sha256((A/'summary.csv').read_bytes()).hexdigest(),'labels':'Color=method; marker=nominal budget; exact star has n evaluations','timing':'Original frozen 3-repeat complete query timing; seed sensitivity timings are excluded'},indent=2))
