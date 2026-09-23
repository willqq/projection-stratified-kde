from sprint_validate import *
configs=json.loads((OUT/'screen_configs.json').read_text())
run_config('matched_projection64',configs['matched_projection64'],(64,128,256),control='single')
s=summarize({k:configs[k] for k in ['b1a_projection64','b1a_projection64_neyman','b0_hamming_J4','matched_projection64']},(64,128,256));s.to_csv(OUT/'finalist_summary.csv',index=False)
print(s.to_string(index=False),flush=True)
