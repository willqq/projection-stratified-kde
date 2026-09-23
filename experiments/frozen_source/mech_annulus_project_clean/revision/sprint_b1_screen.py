import json
from sprint_validate import *
configs=json.loads((OUT/'screen_configs.json').read_text())
configs['matched_Hquarter']['control']='single'
base=dict(score='projection64',J=8,H_fraction=.25,grouping='quantile',full_target=True,allocation='proportional')
for key,cfg in [('b1a_projection64',base),('b1a_projection64_neyman',dict(base,allocation='neyman')),('matched_projection64',dict(base,control='single'))]:
    configs[key]=cfg;run_config(key,cfg,control=cfg.get('control'))
s=summarize(configs);s.to_csv(OUT/'screen_summary.csv',index=False);save_json(OUT/'screen_configs.json',configs)
print(s.pivot(index='config',columns='dataset',values='mean_relative_error').round(4).to_string(),flush=True)
print(s.groupby('config').recovery.median().sort_values(ascending=False).to_string(),flush=True)
