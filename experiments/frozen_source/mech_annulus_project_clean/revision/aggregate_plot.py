"""Recreate all paper numbers, tables and plots from saved raw CSVs only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
ap.add_argument('--out',type=Path)
args=ap.parse_args();b=args.root;out=args.out or b/'results/paper_v1';out.mkdir(parents=True,exist_ok=True)
names={'isolet':'ISOLET','cifar10':'CIFAR-10-Small','cifar10_gist512':'GIST-512','amazon':'Amazon'}
labels={'uniform_mc':'Uniform MC','exact_annular':'Exact annular','approx_annular_mech':'Approx. annular','deann':'DEANN'}
colors={'uniform_mc':'#66717e','exact_annular':'#2976aa','approx_annular_mech':'#c65032','deann':'#16846d'}
markers={'uniform_mc':'o','exact_annular':'s','approx_annular_mech':'^','deann':'D'}
raw=[];input_files=[]
for dataset in names:
    for folder,suffix in [('basic_v1',''),('deann_v1','_deann')]:
        p=b/'results'/folder/f'{dataset}{suffix}_queries.csv'
        input_files.append(p);raw.append(pd.read_csv(p))
frame=pd.concat(raw,ignore_index=True)
# Mean errors weight all 200 fixed queries equally; five sampling seeds each.
summary=frame.groupby(['dataset','method','budget'],as_index=False).agg(
    mare=('relative_error','mean'),mae=('absolute_error','mean'),signed_relative_error=('signed_relative_error','mean'),
    mean_ms=('time_ns',lambda x:x.mean()/1e6),median_ms=('time_ns',lambda x:x.median()/1e6),p95_ms=('time_ns',lambda x:x.quantile(.95)/1e6),
    actual_samples=('actual_samples','mean'),kernel_evaluations=('kernel_evaluations','mean'),distance_evaluations=('distance_evaluations','mean'),
    candidate_size=('candidate_size','mean'),hash_id_checks=('hash_id_checks','mean'),code_distance_checks=('code_distance_checks','mean'),
    encoding_ms=('encoding_ns',lambda x:x.mean()/1e6),candidate_ms=('candidate_ns',lambda x:x.mean()/1e6),
    allocation_ms=('allocation_ns',lambda x:x.mean()/1e6),kernel_ms=('kernel_ns',lambda x:x.mean()/1e6),
    zero_estimates=('zero_estimate','sum'),truth_underflows=('truth_float_underflow','sum'),observations=('query_id','size'))
summary.to_csv(out/'all_budget_summary.csv',index=False)
# Preserve distinct query and sampling variability; no pseudo-independent CI.
frame.groupby(['dataset','method','budget','query_id'],as_index=False).agg(mean_relative_error=('relative_error','mean'),sampling_sd=('relative_error','std')).to_csv(out/'query_seed_summary.csv',index=False)
frame.groupby(['dataset','method','budget','seed'],as_index=False).agg(mare=('relative_error','mean')).to_csv(out/'seed_summary.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
for view in ['budget','time']:
    fig,axes=plt.subplots(2,2,figsize=(7.0,3.4))
    for ax,(dataset,title) in zip(axes.flat,names.items()):
        sub=summary[summary.dataset==dataset]
        for method,label in labels.items():
            data=sub[sub.method==method].sort_values('budget')
            ax.plot(data.budget if view=='budget' else data.median_ms,data.mare,marker=markers[method],color=colors[method],label=label,lw=1.25,ms=3.5)
        ax.set_title(title)
        ax.set_yscale('log');ax.set_ylabel('Mean relative error')
        if view=='budget':
            ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256,512],labels=['32','64','128','256','512']);ax.set_xlabel('Nominal sampling budget M')
        else:
            ax.set_xscale('log');ax.set_xlabel('Complete query median (ms)')
            exact=sub[sub.method=='exact'].iloc[0]
            ax.axvline(exact.median_ms,color='#8d6cad',ls=':',lw=1.2)
        ax.grid(alpha=.2,which='major')
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.005),handlelength=1.7,columnspacing=1.2)
    fig.subplots_adjust(left=.095,right=.985,bottom=.135,top=.865,hspace=.77,wspace=.31)
    fig.savefig(out/f'{view}_error.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/f'{view}_error.png',dpi=180)
    plt.close(fig)
work=summary[(summary.budget==128)|(summary.method=='exact')].copy();work.to_csv(out/'working_point_M128.csv',index=False)
tex=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Dataset & Uniform MC & Exact annular & Approx. annular & DEANN & Exact time \\',r'\midrule']
for dataset,title in names.items():
    sub=work[work.dataset==dataset].set_index('method')
    cells=[title]
    for method in labels:
        v=sub.loc[method];cells.append(f'{100*v.mare:.1f} / {v.median_ms:.3f}')
    cells.append(f'{sub.loc["exact","median_ms"]:.3f}')
    tex.append(' & '.join(cells)+r' \\')
tex += [r'\bottomrule',r'\end{tabular}']
(out/'working_point_M128.tex').write_text('\n'.join(tex)+'\n')
# Predeclared thresholds: describe attainment of each fixed point; do not select a test-optimal point.
th=[]
for row in summary[summary.method!='exact'].itertuples():
    for threshold in [.01,.05,.10]:
        th.append({'dataset':row.dataset,'method':row.method,'budget':row.budget,'threshold':threshold,'attained':bool(row.mare<=threshold),'mare':row.mare,'median_ms':row.median_ms})
pd.DataFrame(th).to_csv(out/'threshold_attainment_all_points.csv',index=False)
mechanism=[];mass=[];mp=[]
for dataset in ['isolet','cifar10_gist512']:
    p=b/'results/mechanism_v1'/f'{dataset}_mechanism_summary.csv';input_files.append(p)
    sm=pd.read_csv(p)
    # Average the five fixed random partitions within a query/budget first.
    s=sm.groupby(['query_id','budget','grouping']).predicted_variance_scaled.mean().unstack('grouping')
    den=s.uniform_on_C
    for grouping in ['approx_annular','random_partition','exact_distance_order']:
        ratio=s[grouping]/den
        for (qi,budget),value in ratio.items():mp.append({'dataset':dataset,'query_id':qi,'budget':budget,'grouping':grouping,'predicted_variance_ratio_to_uniform_C':value})
    rec={'dataset':dataset,'approx_ratio_median':float((s.approx_annular/den).median()),'random_ratio_median':float((s.random_partition/den).median()),'ordered_ratio_median':float((s.exact_distance_order/den).median()),'approx_fraction_below_uniform':float((s.approx_annular<den).mean()),'empirical_prediction_ratio_median':float(sm.variance_ratio.median()),'max_abs_mean_z':float(sm.mean_z.abs().max()),'cases':len(sm)}
    mechanism.append(rec)
    p=b/'results/mechanism_v1'/f'{dataset}_decomposition.csv';input_files.append(p)
    dc=pd.read_csv(p)
    for key in ['candidate','truncated','missing','extra','tail','signed_candidate_discrepancy']:
        recname=key+'_fraction_of_full'
        dc[recname]=dc[key+'_scaled']/dc.full_scaled
    dc['candidate_full_relative_gap']=(dc.candidate_scaled-dc.full_scaled).abs()/dc.full_scaled
    dc.to_csv(out/f'{dataset}_mass_fractions.csv',index=False)
    mass.append({'dataset':dataset,**{col+'_mean':float(dc[col].mean()) for col in dc if col.endswith('_fraction_of_full') or col=='candidate_full_relative_gap'},'candidate_size_mean':float(dc.candidate_size.mean())})
pd.DataFrame(mechanism).to_csv(out/'mechanism_table.csv',index=False)
ratios=pd.DataFrame(mp);ratios.to_csv(out/'mechanism_variance_ratios.csv',index=False)
pd.DataFrame(mass).to_csv(out/'mass_decomposition_summary.csv',index=False)
tex=[r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Dataset & Approx. & Random & Ordered \\',r'\midrule']
for v in mechanism:tex.append(f'{names[v["dataset"]]} & {v["approx_ratio_median"]:.3f} & {v["random_ratio_median"]:.3f} & {v["ordered_ratio_median"]:.3f}'+r' \\')
tex += [r'\bottomrule',r'\end{tabular}'];(out/'mechanism_table.tex').write_text('\n'.join(tex)+'\n')
fig,axes=plt.subplots(1,2,figsize=(7,2.5))
for ax,dataset in zip(axes,['isolet','cifar10_gist512']):
    sub=ratios[ratios.dataset==dataset]
    for j,grouping in enumerate(['approx_annular','random_partition','exact_distance_order']):
        vals=sub[sub.grouping==grouping].predicted_variance_ratio_to_uniform_C.to_numpy()
        offset=(np.arange(len(vals))%11-5)*.025
        ax.scatter(j+offset,vals,s=6,alpha=.45,color=['#c65032','#66717e','#2976aa'][j])
        ax.plot([j-.18,j+.18],[np.median(vals)]*2,color='black',lw=1.6)
    ax.axhline(1,color='black',ls=':',lw=.8);ax.set_yscale('log');ax.set_xticks([0,1,2],labels=['Approx.','Random','Ordered']);ax.set_title(names[dataset]);ax.set_ylabel('Predicted variance / uniform C');ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(out/'mechanism_variance.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(out/'mechanism_variance.png',dpi=180);plt.close(fig)
offline=[]
for dataset in names:
    meta=json.loads((b/'assets'/dataset/'metadata.json').read_text());base=json.loads((b/'results/basic_v1'/f'{dataset}_offline.json').read_text());de=json.loads((b/'results/deann_v1'/f'{dataset}_deann_offline.json').read_text())
    fit=pd.read_csv(b/'results/deann_v1'/f'{dataset}_deann_offline_fits.csv')
    v=work[(work.dataset==dataset)&(work.method=='approx_annular_mech')].iloc[0]
    exact=work[(work.dataset==dataset)&(work.method=='exact')].iloc[0]
    cost=base['train_seconds']+base['build_seconds']
    saving=(exact.mean_ms-v.mean_ms)/1000
    offline.append({'dataset':dataset,**base,**de,'validation_calibration_seconds':meta['validation_calibration_seconds'],'deann_fit_mean_seconds':float(fit.permutation_fit_seconds.mean()),'queries_Q':200,'approx_amortized_mean_ms_at_Q200':v.mean_ms+1000*cost/200,'exact_mean_ms':exact.mean_ms,'break_even_queries':int(np.ceil(cost/saving)) if saving>0 else None,'break_even_status':'finite' if saving>0 else 'none: online mean already slower','raw_rows':meta['raw_shape'][0],'d':meta['raw_shape'][1],'reference_n':meta['split_counts']['reference'],'train_n':meta['split_counts']['train'],'validation_n':80,'test_n':200,'h':meta['bandwidth'],'R':meta['radius'],'delta':meta['delta']})
pd.DataFrame(offline).to_csv(out/'offline_costs_and_protocol.csv',index=False)
provenance={str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
(out/'plot_input_hashes.json').write_text(json.dumps(provenance,indent=2))
(out/'aggregation_notes.json').write_text(json.dumps({'errors':'Mean over 200 fixed test queries and 5 sampling seeds. Exact uses 200 queries once. No confidence intervals claimed.','timing':'Per-row median of three full calls; summary mean/median/P95 of those row medians. Basic repeat uses identical sampled IDs; official DEANN repeats advance the permutation, first repeat provides error.','budget':'DEANN nominal k+m; correction visits can exceed M; actual kernel and ANN distance counts separately saved.','mechanism':'60 fixed query-budget pairs per dataset. Ratio of predicted conditional variances on the same C. Five random partitions averaged before taking ratios and their median. No extra query selection.','amortization':'Q=200, measured means; MECH train plus index build cost, validation calibration listed separately. No amortized saving where online mean is slower.','time_plot':'Dotted vertical line denotes full exact-query median; its error is zero and is not placed on logarithmic error axis.'},indent=2))
print(work[['dataset','method','mare','median_ms']].to_string(index=False))
print(pd.DataFrame(mechanism).to_string(index=False))
print(pd.DataFrame(mass).to_string(index=False))
