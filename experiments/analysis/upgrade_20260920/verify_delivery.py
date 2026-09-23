from pathlib import Path
import json,hashlib,subprocess,re,tempfile,shutil,difflib
import pandas as pd
import pdfplumber
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]; P=R/'manuscript'; Q=R/'qa'; Q.mkdir(exist_ok=True)
checks={}
def check(k,v,detail=None): checks[k]={'pass':bool(v),'details':detail}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'BACKUP_MANIFEST.json').read_text())['files']
bad=[]
for f in manifest:
 for key in ['source','backup']:
  p=Path(f[key])
  if not p.is_file() or sha(p)!=f['sha256']: bad.append(str(p))
check('624_backup_files_unchanged',len(manifest)==624 and not any('/baseline_backup/' in p for p in bad),bad)
check('baseline_research_sources_unchanged',not any(Path(p).name!='.DS_Store' for p in bad),
      {'research_changes': [p for p in bad if Path(p).name!='.DS_Store'],
       'finder_metadata_only_changes': [p for p in bad if Path(p).name=='.DS_Store']})
B=R/'baseline_backup/final_verification_20260920/manuscript'
protected=['table1_final.tex','table2_final.tex','fig3_latency.pdf','fig3_latency.png','references.bib','spconf.sty','IEEEbib.bst','authors.tex']
protected += [str(p.relative_to(B)) for p in (B/'figures').rglob('*') if p.is_file()]
check('frozen_tables_figures_references_style_authors_unchanged',all(sha(P/f)==sha(B/f) for f in protected),protected)
name='ICASSP2027_EN_revised_upgrade'; tex=(P/(name+'.tex')).read_text(); original=(B/'ICASSP2027_EN_revised_final.tex').read_text()
with tempfile.TemporaryDirectory(prefix='icassp-manuscript-verify-') as tmp:
 t=Path(tmp); shutil.copytree(B,t/'baseline_backup/final_verification_20260920/manuscript'); shutil.copytree(R/'scripts',t/'scripts')
 subprocess.run(['python3',str(t/'scripts/update_manuscript.py')],check=True,capture_output=True)
 check('authoring_script_reproduces_final_tex',(t/'manuscript'/(name+'.tex')).read_bytes()==(P/(name+'.tex')).read_bytes())
 shutil.copy2(t/'MANUSCRIPT_CHANGE_RECORD.json',R/'MANUSCRIPT_CHANGE_RECORD.json')
(R/'MANUSCRIPT.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),tex.splitlines(True),fromfile='frozen/ICASSP2027_EN_revised_final.tex',tofile='upgrade/'+name+'.tex')))
abstract=tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
check('abstract_100_to_150_words',100<=len(abstract.split())<=150,len(abstract.split()))
log=(P/(name+'.log')).read_text()
check('no_overfull_undefined_missing_character',not re.search(r'Overfull|undefined|Missing character',log))
fonts=subprocess.check_output(['pdffonts',str(P/(name+'.pdf'))],text=True); (Q/'fonts.txt').write_text(fonts)
check('all_fonts_embedded',all(re.search(r'\s+yes\s+yes\s+',line) for line in fonts.splitlines()[2:]),fonts)
check('no_type3_fonts','Type 3' not in fonts)
with pdfplumber.open(P/(name+'.pdf')) as pdf:
 texts=[p.extract_text() or '' for p in pdf.pages]; (Q/'final_text.txt').write_text('\n\f\n'.join(texts))
 check('five_letter_pages',len(pdf.pages)==5 and all(p.width==612 and p.height==792 for p in pdf.pages))
 check('page4_conclusion_page5_references_only','CONCLUSION' in texts[3] and texts[4].startswith('6. REFERENCES') and not any(x in texts[4] for x in ['CONCLUSION','EXPERIMENTS','Acknowledgment']))
 page_fonts=[]
 for i,p in enumerate(pdf.pages,1):
  chars=[c for c in p.chars if c['text'].strip()]
  page_fonts.append({'page':i,'min_pdf_bp':min(c['size'] if c['upright'] else c['width'] for c in chars)})
 check('extractable_text_at_least_9_TeX_pt',all(x['min_pdf_bp']>=9*72/72.27-.002 for x in page_fonts),page_fonts)
check('17_cited_references',(P/(name+'.bbl')).read_text().count(r'\bibitem')==17)
A=R/'results/formal_analysis'
time=pd.read_csv(A/'path_a_timing_bootstrap.csv'); z=time.query('n==20000').sort_values('budget')
check('A_three_budgets_full_cost_40_4_to_42_9',list(z.budget)==[64,128,256] and list((z.relative_reduction*100).round(1))==[42.9,42.5,40.4] and (z.ci_low>0).all(),z.to_dict('records'))
small=time.query('n<20000')
check('A_small_sizes_full_range',len(small)==35 and round(100*small.relative_reduction.min(),1)==-4.4 and round(100*small.relative_reduction.max(),1)==.3)
summ=pd.read_csv(A/'summary.csv'); timing=summ.query('n==20000 and budget==128').set_index('method')
check('A_M128_1_801_to_1_036',round(timing.loc['projected_current','median_latency_ms'],3)==1.801 and round(timing.loc['projected_fast','median_latency_ms'],3)==1.036)
profile=pd.read_csv(A/'current_profile_mean_ms.csv').query('n==20000').iloc[0]
check('sorting_alone_74_9_percent',round(100*profile.original_sort_mean_ms/profile.profile_total_mean_ms,1)==74.9)
ann=pd.read_csv(A/'ann_H_paired_bootstrap.csv'); tt=ann.query('phase=="test" and budget==128 and anchor!="T"')
check('B_22_2_to_33_5_percent_fixed_ANN_H',len(tt)==4 and round(tt.relative_reduction.min()*100,1)==22.2 and round(tt.relative_reduction.max()*100,1)==33.5)
check('B_69_of70_positive_pointwise_CI',len(ann)==70 and sum(ann.ci_low>0)==69)
nd=pd.read_csv(A/'nondominance_all_points.csv'); multi=nd[nd.method.str.startswith('ann_')&nd.method.str.endswith('_multi')]
check('B_all70_dominated',len(multi)==70 and multi.dominated_by_existing.all())
check('merge_gate_A',json.loads((A/'MERGE_GATE.json').read_text())['decision']=='A')
ver={}
for phase in ['validation','validation_gist20000','test','audit','audit_gist20000']:
 d=json.loads((R/'remote_metadata/upgrade_results'/phase/'VERIFICATION.json').read_text())
 ver[phase]={'passed':d['passed'],'rows':sum(v['rows'] for v in d['datasets'].values())}
check('all_raw_run_verifiers_pass',all(v['passed'] for v in ver.values()),ver)
check('main_new_claims_present',all(s in tex for s in ['40.4--42.9','20,000 GIST','1.801 to 1.036','74.9','4.4']))
# A visual index for the complete supplemental plot collection.
plots=sorted((R/'figures').glob('*.png')); sheet=Image.new('RGB',(1600,450*((len(plots)+2)//3)),'white'); draw=ImageDraw.Draw(sheet)
for i,p in enumerate(plots):
 im=Image.open(p); im.thumbnail((526,415)); x=(i%3)*533; y=(i//3)*450
 sheet.paste(im,(x,y+25)); draw.text((x+4,y+3),p.stem,fill='black')
sheet.save(Q/'figure_contact_sheet.png')
check('12_supplement_plot_pairs',len(plots)==12 and len(list((R/'figures').glob('*.pdf')))==12)
result={'all_checks_pass':all(x['pass'] for x in checks.values()),'checks':checks,'final_tex_sha256':sha(P/(name+'.tex')),'final_pdf_sha256':sha(P/(name+'.pdf')),'authors':'Blank as in frozen baseline; author completion required before upload','visual_review':'Five final manuscript pages inspected; supplemental figures inspected separately'}
(Q/'DELIVERY_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'passed':result['all_checks_pass'],'failed':[k for k,v in checks.items() if not v['pass']],'abstract_words':len(abstract.split()),'rows':ver},ensure_ascii=False))
assert result['all_checks_pass']
