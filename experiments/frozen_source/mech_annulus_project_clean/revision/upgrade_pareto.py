"""Descriptive dominance of measured points; never a significance test."""
import argparse
from pathlib import Path
import pandas as pd


def run(folder):
    data=pd.read_csv(folder/'summary.csv')
    rows=[]
    for keys,g in data.groupby(['dataset','phase','n']):
        exact=g[g.method=='exact'].iloc[0].copy()
        exact['budget']=0
        exact['mean_relative_error']=0.
        exact['median_latency_ms']=g[g.method=='exact'].median_latency_ms.median()
        points=pd.concat([g[g.method!='exact'],pd.DataFrame([exact])],ignore_index=True)
        for _,point in points.iterrows():
            candidates=points[(points.mean_relative_error<=point.mean_relative_error+1e-12)&
                              (points.median_latency_ms<=point.median_latency_ms+1e-12)&
                              ((points.mean_relative_error<point.mean_relative_error-1e-12)|
                               (points.median_latency_ms<point.median_latency_ms-1e-12))]
            original=candidates[~candidates.method.str.startswith('ann_')]
            row=dict(dataset=keys[0],phase=keys[1],n=keys[2],method=point.method,
                budget=int(point.budget),error=point.mean_relative_error,
                latency_ms=point.median_latency_ms,dominated_by_any=bool(len(candidates)),
                dominated_by_existing=bool(len(original)))
            for label,choices in [('any',candidates),('existing',original)]:
                best=choices.sort_values('median_latency_ms').iloc[0] if len(choices) else None
                row[label+'_dominator_method']=best.method if best is not None else None
                row[label+'_dominator_budget']=int(best.budget) if best is not None else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(folder/'nondominance_all_points.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    run(p.parse_args().folder)
