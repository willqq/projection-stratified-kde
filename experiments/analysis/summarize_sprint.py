"""Read immutable query records; aggregate by query and render the existing figures.

No parameter selection is performed here. Bootstrap resamples queries, with the
five sampling seeds averaged within each query. All tested operating points stay.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'; OUT.mkdir(exist_ok=True)
DATASETS=['isolet','cifar10','cifar10_gist512','amazon']
NAMES=dict(zip(DATASETS,['ISOLET','CIFAR-10-Small','GIST-512','Amazon']))
BUDGETS=[32,64,128,256,512]
MAIN=['uniform_mc','exact_annular','final','deann','exact']
COLS=['dataset','phase','method','query_id','raw_row_id','budget','seed',
      'truth','log_truth','estimate','log_estimate','relative_error',
      'signed_relative_error','actual_samples','kernel_evaluations',
      'full_dimension_query_reference_distances','full_dimension_distance_or_dot_evaluations',
      'low_dim_score_evaluations','low_dim_score_dimension','time_ns','transform_ns',
      'candidate_mass_recall','candidate_union_discrepancy','H_count','H_kernel_mass_recall']


def read_queries():
    frames=[]
    for folder in sorted((ROOT/'results/formal').glob('*/*')):
        if not (folder/'DONE.json').exists():raise RuntimeError('Incomplete formal result '+str(folder))
        for filename in ['basic_queries.csv','deann_queries.csv']:
            frames.append(pd.read_csv(folder/filename,usecols=lambda x:x in COLS))
    f=pd.concat(frames,ignore_index=True)
    if len(f)!=246400:raise RuntimeError(f'Expected 246400 observations, got {len(f)}')
    keys=['phase','dataset','method','budget','seed','query_id']
    assert not f.duplicated(keys).any()
    # Proportional control is exactly the frozen final method, explicitly aliased.
    alias=f[f.method=='final'].copy();alias['method']='matched_proportional'
    return pd.concat([f,alias],ignore_index=True)


def aggregate(f):
    q=f.groupby(['phase','dataset','method','budget','query_id','raw_row_id'],as_index=False).agg(
        mean_error=('relative_error','mean'),median_latency_ns=('time_ns','median'),
        kernel_evaluations=('kernel_evaluations','mean'),actual_samples=('actual_samples','mean'))
    rows=[]
    for key,g in f.groupby(['phase','dataset','method','budget']):
        phase,dataset,method,M=key
        qq=q[(q.phase==phase)&(q.dataset==dataset)&(q.method==method)&(q.budget==M)]
        rows.append(dict(phase=phase,dataset=dataset,method=method,budget=int(M),
            mean_relative_error=float(g.relative_error.mean()),
            median_query_mean_error=float(qq.mean_error.median()),
            p90_query_mean_error=float(qq.mean_error.quantile(.9)),
            se_query_cluster=float(qq.mean_error.std(ddof=1)/np.sqrt(len(qq))),
            median_latency_ms=float(g.time_ns.median()/1e6),
            p90_latency_ms=float(g.time_ns.quantile(.9)/1e6),
            mean_actual_samples=float(g.actual_samples.mean()),
            mean_kernel_evaluations=float(g.kernel_evaluations.mean()),
            mean_full_dimension_distances_or_dots=float(g.full_dimension_distance_or_dot_evaluations.mean()),
            low_dim_score_evaluations=float(g.low_dim_score_evaluations.mean()),
            observations=len(g),queries=len(qq)))
    s=pd.DataFrame(rows);s.to_csv(OUT/'summary.csv',index=False)
    q.to_csv(OUT/'query_cluster_means.csv',index=False)
    return s,q


def bootstrap(q):
    rng=np.random.default_rng(20260916);recovery=[];contrasts=[]
    for (phase,dataset,M),g in q[q.budget>0].groupby(['phase','dataset','budget']):
        w=g.pivot(index='raw_row_id',columns='method',values='mean_error').sort_index()
        assert len(w)==200 and not w.isna().any().any()
        ids=rng.integers(0,len(w),size=(1000,len(w)))
        u=w.uniform_mc.to_numpy();x=w.exact_annular.to_numpy();a=w.final.to_numpy()
        den=u.mean()-x.mean();denboot=(u-x)[ids].mean(axis=1)
        rec=(u.mean()-a.mean())/den if den>max(1e-8,.01*u.mean()) else np.nan
        stable=bool(np.quantile(denboot,.025)>0)
        rb=(u-a)[ids].mean(axis=1)/denboot
        recovery.append(dict(phase=phase,dataset=dataset,budget=M,recovery=rec,
            ci_low=float(np.quantile(rb,.025)) if stable else np.nan,
            ci_high=float(np.quantile(rb,.975)) if stable else np.nan,
            denominator=den,denominator_ci_low=float(np.quantile(denboot,.025)),
            denominator_ci_high=float(np.quantile(denboot,.975)),denominator_stable=stable,
            reduction_vs_MC=1-a.mean()/u.mean(),exact_reduction_vs_MC=1-x.mean()/u.mean()))
        for other in ['current_approx','uniform_mc','deann','matched_single','matched_neyman']:
            b=w[other].to_numpy();delta=a-b;db=delta[ids].mean(axis=1)
            ab=a[ids].mean(axis=1);bb=b[ids].mean(axis=1);rr=1-ab/bb
            contrasts.append(dict(phase=phase,dataset=dataset,budget=M,method='final',comparator=other,
                mean_error_difference=delta.mean(),difference_ci_low=np.quantile(db,.025),difference_ci_high=np.quantile(db,.975),
                relative_error_reduction=1-a.mean()/b.mean(),reduction_ci_low=np.quantile(rr,.025),reduction_ci_high=np.quantile(rr,.975),
                query_count=len(w),bootstrap_resamples=1000,bootstrap_seed=20260916))
    rec=pd.DataFrame(recovery);con=pd.DataFrame(contrasts)
    rec.to_csv(OUT/'recovery.csv',index=False);con.to_csv(OUT/'paired_bootstrap.csv',index=False)
    return rec,con


def pareto(s):
    rows=[]
    for (phase,dataset),allg in s.groupby(['phase','dataset']):
        for _,a in allg[allg.method=='final'].iterrows():
            def dominators(g):
                return g[(g.mean_relative_error<=a.mean_relative_error)&(g.median_latency_ms<=a.median_latency_ms)&
                         ((g.mean_relative_error<a.mean_relative_error)|(g.median_latency_ms<a.median_latency_ms))]
            de=dominators(allg[allg.method=='deann']);main=dominators(allg[allg.method.isin(MAIN)])
            expanded=dominators(allg[allg.method!='current_approx'])
            ids=lambda g:json.dumps([str(r.method)+':M'+str(int(r.budget)) for _,r in g.iterrows()])
            rows.append(dict(phase=phase,dataset=dataset,budget=int(a.budget),
                deann_nondominated=len(de)==0,main_frontier=len(main)==0,including_controls_frontier=len(expanded)==0,
                deann_dominators=ids(de),main_dominators=ids(main),including_controls_dominators=ids(expanded)))
    p=pd.DataFrame(rows);p.to_csv(OUT/'pareto_all_points.csv',index=False)
    return p


def fixed_variance():
    fs=[]
    for path in sorted((ROOT/'results/fixed_c_final').glob('*/*_variance.csv')):
        fs.append(pd.read_csv(path))
    v=pd.concat(fs,ignore_index=True)
    q=v.groupby(['phase','dataset','query_id','grouping','budget'],as_index=False).variance_ratio.mean()
    rows=[];rng=np.random.default_rng(20260917)
    for (phase,dataset,M),g in q.groupby(['phase','dataset','budget']):
        w=g.pivot(index='query_id',columns='grouping',values='variance_ratio').sort_index()
        ids=rng.integers(0,len(w),size=(1000,len(w)))
        for group in ['approx','random','ordered']:
            vals=w[group].to_numpy();bs=np.median(vals[ids],axis=1)
            rows.append(dict(phase=phase,dataset=dataset,budget=int(M),grouping=group,
                median_variance_ratio=float(np.median(vals)),mean_variance_ratio=float(np.mean(vals)),
                median_ci_low=float(np.quantile(bs,.025)),median_ci_high=float(np.quantile(bs,.975)),
                ratio_to_matched_random_median=float(np.median(vals/w.random.to_numpy())),queries=len(w)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'fixed_c_variance.csv',index=False)
    return result


def render_fig3(s,phase='test'):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,'axes.labelsize':9.2,'axes.titlesize':9.2,
        'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.linewidth':.6,'lines.linewidth':1.25})
    fig,axes=plt.subplots(2,2,figsize=(7,2.85));fig.subplots_adjust(left=.085,right=.99,bottom=.19,top=.83,hspace=.5,wspace=.3)
    methods=['uniform_mc','exact_annular','final','deann']
    labels=['Uniform MC','Exact annular','Projected strata','DEANN']
    colors=['#6A7077','#2878B5','#C13F30','#26946A'];markers=['o','s','D','^'];handles=[]
    for ax,ds in zip(axes.flat,DATASETS):
        if phase=='audit' and ds=='amazon':
            ax.text(.5,.5,'No independent audit rows',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
        g=s[(s.phase==phase)&(s.dataset==ds)]
        for i,m in enumerate(methods):
            z=g[g.method==m].sort_values('budget')
            line,=ax.plot(z.budget,z.mean_relative_error*100,color=colors[i],marker=markers[i],markersize=3.1,label=labels[i])
            if ds==DATASETS[0]:handles.append(line)
        ax.set_xscale('log',base=2);ax.set_xticks(BUDGETS,[str(x) for x in BUDGETS]);ax.minorticks_off()
        ax.set_title(NAMES[ds],pad=3);ax.grid(axis='y',alpha=.22,linewidth=.5);ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'Nominal sampling budget M',ha='center',fontsize=9.2)
    fig.text(.012,.48,'Mean relative error (%)',rotation=90,va='center',fontsize=9.2)
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.53,1.015),handlelength=1.55,columnspacing=1)
    stem='fig3_final' if phase=='test' else 'fig3_audit_supplement'
    fig.savefig(ROOT/(stem+'.pdf'));fig.savefig(ROOT/(stem+'.png'),dpi=220);plt.close(fig)


def tables(s,v):
    g=s[(s.phase=='test')&(s.budget==128)];exact=s[(s.phase=='test')&(s.method=='exact')]
    lines=[r'\begin{tabular}{lccccc}',r'\toprule',r'Dataset & Uniform MC & Exact annular & Projected strata & DEANN & Exact time \\',r'\midrule']
    for ds in DATASETS:
        cells=[]
        for method in ['uniform_mc','exact_annular','final','deann']:
            z=g[(g.dataset==ds)&(g.method==method)].iloc[0]
            cells.append(f'{100*z.mean_relative_error:.1f} / {z.median_latency_ms:.3f}')
        t=exact[exact.dataset==ds].iloc[0].median_latency_ms
        lines.append(NAMES[ds]+' & '+' & '.join(cells)+f' & {t:.3f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(ROOT/'table1_final.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lcccccc}',r'\toprule',r'& \multicolumn{3}{c}{A: fixed-$C$ variance ratio} & \multicolumn{3}{c}{B: matched-$H$ error (\%)} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Dataset & Projected & Random & Ordered & Single & Multi & Neyman \\',r'\midrule']
    for ds in DATASETS:
        cells=[]
        for group in ['approx','random','ordered']:
            z=v[(v.phase=='test')&(v.dataset==ds)&(v.budget==128)&(v.grouping==group)].iloc[0]
            cells.append(f'{z.median_variance_ratio:.3f}')
        for m in ['matched_single','final','matched_neyman']:
            z=g[(g.dataset==ds)&(g.method==m)].iloc[0];cells.append(f'{100*z.mean_relative_error:.1f}')
        lines.append(NAMES[ds]+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(ROOT/'table2_final.tex').write_text('\n'.join(lines)+'\n')


def main():
    f=read_queries();s,q=aggregate(f);rec,con=bootstrap(q);p=pareto(s);v=fixed_variance()
    render_fig3(s);render_fig3(s,'audit');tables(s,v)
    print(s[(s.budget==128)&s.method.isin(MAIN)][['phase','dataset','method','mean_relative_error','median_latency_ms']].to_string(index=False))
    print(rec[rec.budget==128].to_string(index=False));print(p.to_string(index=False))
    (OUT/'AGGREGATION_NOTES.md').write_text('All means retain every recorded query and sampling seed. Standard errors and bootstrap intervals use query clusters after averaging five seeds. Latency is the median of per-observation three-call medians. Per-phase medians are not additive. DEANN counts in summary refer to the first call; raw timing rows preserve operation counts for each call. Frontier labels compare all recorded budgets without interpolation and are limited to measured points. Fixed-C Table2A has H=0; Table2B instead matches H=floor(M/4). Exact is one observation per query with budget=0. No test or audit output feeds parameter selection.\n')

if __name__=='__main__':main()
