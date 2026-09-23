from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; A=ROOT/'analysis';A.mkdir(exist_ok=True)
LABEL={'isolet':'ISOLET','cifar10':'CIFAR-10','cifar10_gist512':'GIST-512','amazon':'Amazon'}
MLABEL={'uniform_mc':'Uniform MC','exact_annular':'Exact annular','final':'Projected strata',
 'exact':'Exact sum','deann_original':'DEANN: original',
 'deann_mean_relative_error':'DEANN: validation-error',
 'deann_median_time_ns':'DEANN: validation-latency',
 'deann_error_time_product':'DEANN: validation-product'}
COLOR={'uniform_mc':'#777777','exact_annular':'#d8861b','final':'#1477b5','exact':'#111111',
 'deann_original':'#579867','deann_mean_relative_error':'#875bb2','deann_median_time_ns':'#be5163','deann_error_time_product':'#886c43'}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})

def bootstrap(p):
    ix=np.random.default_rng(20260920).integers(0,len(p),size=(1000,len(p)))
    return p.to_numpy()[ix].mean(axis=1),list(p.columns)

def bandwidth():
    frames=[];var=[]
    for f in sorted((R/'bandwidth').glob('*_queries.csv')):frames.append(pd.read_csv(f))
    for f in sorted((R/'bandwidth').glob('*_variance.csv')):var.append(pd.read_csv(f))
    if not frames:return
    raw=pd.concat(frames,ignore_index=True);v=pd.concat(var,ignore_index=True)
    q=raw.groupby(['dataset','phase','h_factor','query_id','method']).relative_error.mean().unstack('method')
    rows=[]
    for (ds,ph,h),p in q.groupby(level=[0,1,2]):
      means=p.mean();b,cols=bootstrap(p);bm={k:b[:,i] for i,k in enumerate(cols)}
      gain=100*(1-means['final']/means['matched_single']);gb=100*(1-bm['final']/bm['matched_single'])
      denom=means['uniform_mc']-means['exact_annular'];db=bm['uniform_mc']-bm['exact_annular']
      rec=(means['uniform_mc']-means['final'])/denom
      rb=(bm['uniform_mc']-bm['final'])/db
      vv=v[(v.dataset==ds)&(v.phase==ph)&(v.h_factor==h)]
      rows.append(dict(dataset=ds,phase=ph,h_factor=h,budget=128,queries=len(p),
         **{k+'_error_percent':100*x for k,x in means.items()},matched_gain_percent=gain,
         matched_gain_ci_low=np.quantile(gb,.025),matched_gain_ci_high=np.quantile(gb,.975),
         variance_ratio_median=vv.variance_ratio.median(),recovery=rec,
         recovery_ci_low=np.quantile(rb,.025),recovery_ci_high=np.quantile(rb,.975),
         recovery_denominator=denom,recovery_denominator_ci_low=np.quantile(db,.025),
         recovery_denominator_ci_high=np.quantile(db,.975),recovery_stable=bool(np.quantile(db,.025)>0)))
    s=pd.DataFrame(rows);s.to_csv(A/'bandwidth_summary.csv',index=False)
    q.to_csv(A/'bandwidth_query_means.csv')
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for ax,(ds,label) in zip(axes.ravel(),LABEL.items()):
      for ph,m,offset in [('test','o',-.015),('audit','s',.015)]:
        z=s[(s.dataset==ds)&(s.phase==ph)].sort_values('h_factor')
        if len(z):ax.errorbar(z.h_factor+offset,z.matched_gain_percent,yerr=[z.matched_gain_percent-z.matched_gain_ci_low,z.matched_gain_ci_high-z.matched_gain_percent],fmt=m+'-',capsize=3,label=ph)
      ax.axhline(0,color='gray',lw=.8);ax.set_title(label);ax.grid(alpha=.2);ax.set_xticks([.5,.75,1,1.5,2]);ax.legend(fontsize=9)
    fig.supxlabel('Bandwidth multiplier h / h0');fig.supylabel('Matched-H relative error reduction (%)')
    fig.tight_layout();fig.savefig(A/'bandwidth_matched_H.png',dpi=180);fig.savefig(A/'bandwidth_matched_H.pdf');plt.close(fig)

def summarize_runs(kind):
    frames=[]
    for f in sorted((R/kind).rglob('*_queries.csv')):
      if kind=='deann_formal':ds,ph=f.parts[-3:-1];extra=dict(dataset=ds,phase=ph)
      else:extra=dict(dataset='cifar10_gist512',phase='audit_prefix100',n=int(f.parent.name))
      z=pd.read_csv(f)
      for k,v in extra.items():z[k]=v
      frames.append(z)
    if not frames:return
    raw=pd.concat(frames,ignore_index=True);ctx=['dataset','phase']+(['n'] if kind=='scaling' else [])
    s=raw.groupby(ctx+['method','budget'],as_index=False).agg(mean_error=('relative_error','mean'),median_latency_ms=('time_ns',lambda x:x.median()/1e6),mean_kernel_evaluations=('kernel_evaluations','mean'),observations=('query_id','size'))
    q=raw.groupby(ctx+['method','budget','query_id']).relative_error.mean().reset_index();q.to_csv(A/f'{kind}_query_means.csv',index=False)
    CIs=[]
    for keys,z in q.groupby(ctx+['method','budget']):
      p=z[['relative_error']];b,_=bootstrap(p)
      CIs.append(dict(zip(ctx+['method','budget'],keys),error_ci_low=np.quantile(b,.025),error_ci_high=np.quantile(b,.975)))
    s=s.merge(pd.DataFrame(CIs));s['mean_error_percent']=100*s.mean_error
    s['nondominated']=False;s['dominators']=''
    for _,z in s.groupby(ctx):
      for idx,row in z.iterrows():
        # Full-dimensional exact numerical zero has no scientific error.
        err=z.mean_error.where(z.method!='exact',0)
        target=0 if row.method=='exact' else row.mean_error
        dom=z[(err<=target)&(z.median_latency_ms<=row.median_latency_ms)&((err<target)|(z.median_latency_ms<row.median_latency_ms))]
        s.loc[idx,'nondominated']=len(dom)==0
        s.loc[idx,'dominators']=';'.join(dom.method+':M'+dom.budget.astype(str))
    s.to_csv(A/f'{kind}_summary.csv',index=False)
    if kind=='deann_formal':
      prior=pd.read_csv(ROOT.parent/'sprint_final/analysis/pareto_all_points.csv')
      change=[]
      for (ds,ph),z in s.groupby(['dataset','phase']):
        original=z[z.method.isin(['final','uniform_mc','exact_annular','exact','deann_original'])]
        for _,f in z[z.method=='final'].iterrows():
          dom=original[(original.mean_error.where(original.method!='exact',0)<=f.mean_error)&(original.median_latency_ms<=f.median_latency_ms)&((original.mean_error.where(original.method!='exact',0)<f.mean_error)|(original.median_latency_ms<f.median_latency_ms))]
          old=prior[(prior.dataset==ds)&(prior.phase==ph)&(prior.budget==f.budget)].iloc[0]
          change.append(dict(dataset=ds,phase=ph,budget=f.budget,frozen_paper_frontier=bool(old.main_frontier),
             rerun_original_configuration_frontier=len(dom)==0,expanded_frontier=bool(f.nondominated),
             original_configuration_dominators=';'.join(dom.method+':M'+dom.budget.astype(str)),expanded_dominators=f.dominators))
      pd.DataFrame(change).to_csv(A/'deann_frontier_change.csv',index=False)
    rec=[]
    for keys,z in q[q.method.isin(['uniform_mc','exact_annular','final'])].groupby(ctx+['budget']):
      p=z.pivot(index='query_id',columns='method',values='relative_error');means=p.mean();b,cols=bootstrap(p);bm={k:b[:,i] for i,k in enumerate(cols)}
      denom=means.uniform_mc-means.exact_annular;db=bm['uniform_mc']-bm['exact_annular']
      rb=(bm['uniform_mc']-bm['final'])/db
      rec.append(dict(zip(ctx+['budget'],keys),recovery=(means.uniform_mc-means.final)/denom,recovery_ci_low=np.quantile(rb,.025),recovery_ci_high=np.quantile(rb,.975),denominator_ci_low=np.quantile(db,.025)))
    pd.DataFrame(rec).to_csv(A/f'{kind}_recovery.csv',index=False)
    pairs=[]
    for keys,z in q[q.budget>0].groupby(ctx+['budget']):
      p=z.pivot(index='query_id',columns='method',values='relative_error')
      b,cols=bootstrap(p);bm={k:b[:,i] for i,k in enumerate(cols)};mu=p.mean()
      for comp in p.columns:
        if comp=='final':continue
        gain=100*(1-mu['final']/mu[comp]);gb=100*(1-bm['final']/bm[comp])
        pairs.append(dict(zip(ctx+['budget'],keys),comparator=comp,
          final_gain_percent=gain,gain_ci_low=np.quantile(gb,.025),gain_ci_high=np.quantile(gb,.975),
          error_difference_percent_points=100*(mu[comp]-mu['final']),
          difference_ci_low=100*np.quantile(bm[comp]-bm['final'],.025),difference_ci_high=100*np.quantile(bm[comp]-bm['final'],.975)))
    pd.DataFrame(pairs).to_csv(A/f'{kind}_paired_comparisons.csv',index=False)
    if kind=='scaling':
      timing=[]
      for f in sorted((R/kind).rglob('basic_timings.csv')):
        z=pd.read_csv(f);z=z[(z.method=='final')&(z.budget==128)].copy();z['n']=int(f.parent.name);timing.append(z)
      t=pd.concat(timing);components=['projection_ns','low_dim_score_ns','sorting_and_strata_ns','kernel_ns'];t['other_ns']=t.time_ns-t[components].sum(axis=1)
      assert (t.other_ns>=0).all()
      g=t.groupby('n')[components+['other_ns','time_ns']].mean()/1e6
      assert np.allclose(g[components+['other_ns']].sum(axis=1),g.time_ns)
      g.to_csv(A/'scaling_breakdown_mean_ms.csv')
      fig,ax=plt.subplots(figsize=(7,4));bottom=np.zeros(len(g))
      for key,label in zip(components+['other_ns'],['Query projection','Low-dimensional scan','Sort + strata','Original-space kernels','Other']):
        ax.bar(np.arange(len(g)),g[key],bottom=bottom,label=label);bottom+=g[key].to_numpy()
      ax.set_xticks(np.arange(len(g)),g.index.astype(str));ax.set_xlabel('Reference size n');ax.set_ylabel('Mean complete-query latency (ms)');ax.legend(fontsize=9)
      fig.tight_layout();fig.savefig(A/'scaling_breakdown.png',dpi=180);fig.savefig(A/'scaling_breakdown.pdf');plt.close(fig)
      fig,axs=plt.subplots(1,2,figsize=(10,4))
      for method in s.method.unique():
        z=s[(s.method==method)&(s.budget.isin([0,128]))].sort_values('n')
        axs[0].plot(z.n,z.mean_error_percent,'o-',label=MLABEL[method],color=COLOR[method]);axs[1].plot(z.n,z.median_latency_ms,'o-',label=MLABEL[method],color=COLOR[method])
      for ax in axs:
        ax.set_xscale('log');ax.set_xlabel('Reference size n');ax.grid(alpha=.2)
        ax.set_xticks([1000,1800,5000,10000,20000],['1k','1.8k','5k','10k','20k']);ax.xaxis.set_minor_formatter(NullFormatter())
      axs[0].set_ylabel('Mean relative error (%)');axs[1].set_ylabel('Median complete latency (ms)');axs[1].legend(fontsize=9)
      fig.tight_layout();fig.savefig(A/'scaling_M128.png',dpi=180);fig.savefig(A/'scaling_M128.pdf');plt.close(fig)
    else:
      fig,axes=plt.subplots(2,2,figsize=(10,7))
      handles={}
      for ax,(ds,label) in zip(axes.ravel(),LABEL.items()):
        z=s[(s.dataset==ds)&(s.phase=='test')]
        for method,zz in z.groupby('method'):
          zz=zz.sort_values('budget');line,=ax.plot(zz.median_latency_ms,zz.mean_error_percent,'-',label=MLABEL[method],color=COLOR[method],lw=2 if method=='final' else 1)
          handles[method]=line
          for _,row in zz.iterrows():
            marker={0:'*',32:'o',64:'^',128:'s',256:'D',512:'p'}[row.budget]
            ax.scatter(row.median_latency_ms,row.mean_error_percent,marker=marker,s=35,color=COLOR[method])
        ax.set_title(label);ax.set_xscale('log');ax.grid(alpha=.2)
        ticks={'isolet':[.05,.1,.2,.4,.7],'cifar10':[.15,.3,.6,1,2],
          'cifar10_gist512':[.05,.1,.2,.4,.6],'amazon':[.4,.8,1.5,3,6]}[ds]
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FormatStrFormatter('%g'));ax.xaxis.set_minor_formatter(NullFormatter())
      fig.supxlabel('Median complete-query latency (ms, log scale)');fig.supylabel('Mean relative error (%)')
      fig.legend(handles.values(),[MLABEL[k] for k in handles],loc='upper center',ncol=4,fontsize=9)
      mh=[Line2D([],[],color='black',marker=m,linestyle='',label='M='+str(b)) for b,m in [(32,'o'),(64,'^'),(128,'s'),(256,'D'),(512,'p')]]
      fig.legend(handles=mh,loc='lower center',bbox_to_anchor=(.5,.038),ncol=5,fontsize=9)
      fig.tight_layout(rect=[.01,.085,1,.925]);fig.savefig(A/'deann_stress_latency.png',dpi=180);fig.savefig(A/'deann_stress_latency.pdf');plt.close(fig)

if __name__=='__main__':
    bandwidth();summarize_runs('deann_formal');summarize_runs('scaling')
