"""Replot the stored error/complete-query-latency measurements, all budgets."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/paper');a=p.parse_args()
df=pd.read_csv(a.out/'latency_points.csv');df=df[df.phase=='test']
methods={'uniform_mc':('Uniform MC','#777777','-'),'exact_annular':('Exact annular','#d8861b','-'),'final':('Projected strata','#1477b5','-'),'deann_original':('DEANN-O','#579867',':'),'deann_mean_relative_error':('DEANN-E','#875bb2','--'),'deann_median_time_ns':('DEANN-T','#be5163','-.')}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(10,6.5))
markers={32:'o',64:'^',128:'s',256:'D',512:'p'}
for ax,(ds,name) in zip(axs.flat,{'isolet':'ISOLET','cifar10':'CIFAR-10-Small','cifar10_gist512':'GIST-512','amazon':'Amazon'}.items()):
    s=df[df.dataset==ds]
    for method,(label,color,style) in methods.items():
        z=s[s.method==method].sort_values('budget')
        if z.empty:continue
        ax.plot(z.latency_ms,z.error*100,color=color,ls=style,label=label)
        for r in z.itertuples():ax.plot(r.latency_ms,r.error*100,marker=markers[int(r.budget)],color=color,ls='')
    z=s[s.method=='exact'];ax.plot(z.latency_ms,0*z.error,marker='*',ls='',color='black')
    ax.set_xscale('log');ax.set_title(name);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
handles=[Line2D([],[],color=c,ls=s,label=l) for l,c,s in methods.values()]+[Line2D([],[],color='black',marker='*',ls='',label='Exact sum')]
fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False)
fig.supxlabel('Median complete-query latency (ms, log scale)',y=.012);fig.supylabel('Mean relative error (%)')
fig.text(.5,.062,'Markers: circle 32; triangle 64; square 128; diamond 256; pentagon 512',ha='center',fontsize=9)
fig.tight_layout(rect=(.02,.07,1,.88));fig.savefig(a.out/'latency.png',dpi=160);fig.savefig(a.out/'latency.pdf')
print(a.out/'latency.png')
