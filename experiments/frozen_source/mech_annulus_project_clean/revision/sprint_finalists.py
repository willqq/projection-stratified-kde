import json
from sprint_validate import *
configs=json.loads((OUT/'screen_configs.json').read_text())
keys=['b1a_projection64','b1a_projection64_neyman','b0_hamming_J4']
save_json(OUT/'shortlist_before_extension.json',dict(methods=keys,controls=['matched_projection64'],budgets=[64,128,256],reason='Three highest median M128 validation Recovery candidate methods; matched control is not a candidate.'))
for key in keys+['matched_projection64']:
    run_config(key,configs[key],(64,128,256),control=configs[key].get('control'))
s=summarize({k:configs[k] for k in keys+['matched_projection64']},(64,128,256));s.to_csv(OUT/'finalist_summary.csv',index=False)
print(s[['config','dataset','budget','mean_relative_error','recovery','median_time_ms']].to_string(index=False),flush=True)
