"""Render all frozen curves without selecting favorable budgets or datasets."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import pandas as pd


LABELS = {'exact': 'Exact sum', 'uniform_mc': 'Uniform MC',
    'exact_annular': 'Exact annular', 'projected_current': 'Projected: full sort',
    'projected_fast': 'Projected: multirank', 'official_O': 'Official DEANN O',
    'official_E': 'Official DEANN E', 'official_T': 'Official DEANN T',
    'ann_O_single': 'ANN E=O: Single', 'ann_O_multi': 'ANN E=O: Multi',
    'ann_E_single': 'ANN E: Single', 'ann_E_multi': 'ANN E: Multi',
    'ann_T_single': 'ANN T: Single', 'ann_T_multi': 'ANN T: Multi'}
NAMES = {'isolet':'ISOLET', 'cifar10':'CIFAR-10-Small',
         'cifar10_gist512':'GIST-512', 'amazon':'Amazon'}
MARKERS = {32:'o',64:'^',128:'s',256:'D',512:'p'}


def run(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    data = pd.read_csv(source / 'summary.csv')
    for (dataset, phase, n), group in data.groupby(['dataset','phase','n']):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        # Reserve explicit space for both legends; constrained_layout accounts
        # for only the most recently attached legend in this two-legend layout.
        fig.subplots_adjust(left=.09, right=.71, bottom=.12, top=.90)
        for index, (method, curve) in enumerate(group.groupby('method', sort=True)):
            curve = curve.sort_values('budget')
            if method == 'exact':
                ax.scatter(curve.median_latency_ms.median(), 0, marker='*', s=100,
                           c='black', label=LABELS[method])
                continue
            style = ':' if method.startswith('official') else '--' if method.endswith('single') else '-'
            line, = ax.plot(curve.median_latency_ms, 100*curve.mean_relative_error,
                linestyle=style, color=plt.cm.tab20(index % 20),
                label=LABELS.get(method,method))
            for row in curve.itertuples():
                ax.scatter(row.median_latency_ms,100*row.mean_relative_error,
                           marker=MARKERS[row.budget],s=28,color=line.get_color())
        ax.set(xlabel='Complete query median (ms)', ylabel='Mean relative error (%)',
               title=f'{NAMES[dataset]}: {phase}, n={n:,}')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
        fig.legend(*ax.get_legend_handles_labels(), loc='upper left',
                   bbox_to_anchor=(.73,.91), frameon=False, fontsize=9)
        fig.legend(handles=[Line2D([],[],color='black',ls='',marker=MARKERS[m],label=f'M={m}')
                           for m in sorted(group.budget.unique())],
                  loc='lower left',bbox_to_anchor=(.73,.10),frameon=False,fontsize=9)
        stem = destination / f'latency_{phase}_{dataset}_n{n}'
        fig.savefig(str(stem)+'.pdf')
        fig.savefig(str(stem)+'.png', dpi=160)
        plt.close(fig)
    timing = pd.read_csv(source / 'path_a_timing_bootstrap.csv')
    for (phase,n), g in timing.groupby(['phase','n']):
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for dataset,z in g.groupby('dataset'):
            z=z.sort_values('budget')
            ax.errorbar(z.budget,100*z.relative_reduction,
                yerr=[100*(z.relative_reduction-z.ci_low),100*(z.ci_high-z.relative_reduction)],
                marker='o',capsize=3,label=NAMES[dataset])
        ax.axhline(0,color='gray',lw=.8)
        ax.axhline(20,color='gray',ls='--',lw=.8,label='Predeclared project threshold')
        ax.set(xlabel='Kernel budget M',ylabel='Full-query time reduction (%)',
               title=f'Equivalent partition: {phase}, n={n:,}')
        ax.legend(frameon=False,fontsize=9)
        ax.grid(alpha=.2)
        stem=destination/f'partition_speed_{phase}_n{n}'
        fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.png',dpi=160)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();run(a.source,a.destination)
