from pathlib import Path
import hashlib,json,re,subprocess,urllib.request
import pdfplumber
import pandas as pd
R=Path(__file__).resolve().parents[1];M=R/'manuscript';Q=R/'qa'
pdf=M/'ICASSP2027_EN_revised_final.pdf'
fontout=subprocess.check_output(['pdffonts',str(pdf)],text=True)
(Q/'pdffonts.txt').write_text(fontout)
report={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'bytes':pdf.stat().st_size,'pages':[]}
with pdfplumber.open(pdf) as doc:
 assert len(doc.pages)==5
 for i,page in enumerate(doc.pages):
  assert (page.width,page.height)==(612,792)
  text=page.extract_text() or ''
  # pdfminer reports glyph height rather than text-matrix font size for rotated labels.
  chars=[c for c in page.chars if c.get('upright',True) and c['text'].strip()]
  small=[(c['text'],c['size']*72.27/72,c['fontname']) for c in chars if c['size']*72.27/72<8.999]
  assert not small,small[:10]
  report['pages'].append({'page':i+1,'min_upright_size_tex_pt':min(c['size']*72.27/72 for c in chars),'characters':len(chars),'text_bbox':[min(c['x0'] for c in chars),min(c['top'] for c in chars),max(c['x1'] for c in chars),max(c['bottom'] for c in chars)]})
  if i==4:
   assert 'REFERENCES' in text and 'CONCLUSION' not in text and 'Table 2' not in text
  (Q/f'page_{i+1}.txt').write_text(text)
 assert not doc.metadata.get('Author')
for line in fontout.splitlines()[2:]:
 assert re.search(r'yes\s+yes\s+(yes|no)\s+\d+\s+\d+',line),line
 assert 'Type 3' not in line
report.update(page_count=5,fonts_embedded=True,type3=False,font_minimum='9 TeX pt; method labels 9.2; chart font settings >=9.2 with actual-scale check',authors='blank inherited; formal metadata pending',technical_pages=4,fifth_page='references only')
log=(M/'ICASSP2027_EN_revised_final.log').read_text()
assert 'Overfull' not in log and 'undefined' not in log.lower()
report['overfull_or_undefined']=False
for file in ['spconf.sty','IEEEbib.bst']:
 url='https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/'+file
 data=urllib.request.urlopen(url,timeout=30).read()
 assert data==(M/file).read_bytes(),file
 report[file]={'official_url':url,'sha256':hashlib.sha256(data).hexdigest()}
base=json.loads((R/'BASE_PROVENANCE.json').read_text())
src=Path(base['source_directory'])
assert all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in base['files'].items())
report['original_main_base_unchanged']=True

s=pd.read_csv(R/'evidence/analysis/summary.csv');rec=pd.read_csv(R/'evidence/analysis/recovery.csv')
paired=pd.read_csv(R/'evidence/analysis/paired_bootstrap.csv')
test=s[(s.phase=='test')&(s.budget>0)].pivot(index=['dataset','budget'],columns='method',values='mean_relative_error')
improve=1-test.exact_annular/test.uniform_mc
assert round(improve.min()*100,1)==51.5 and round(improve.max()*100,1)==71.9
for phase,low,high in [('test',17.7,24.5),('audit',21.7,22.3)]:
 b=paired[(paired.phase==phase)&(paired.budget==128)&(paired.method=='final')&(paired.comparator=='matched_single')]
 assert round(b.relative_error_reduction.min()*100,1)==low
 assert round(b.relative_error_reduction.max()*100,1)==high
 assert (b.reduction_ci_low>0).all()
v=pd.read_csv(R/'evidence/isolet_summary.csv');vv=v[(v.kind=='hamming')&(v.J==8)&(v.grouping=='old_annuli')&(v.budget==128)]
assert round(vv.variance_ratio_median.iloc[0],3)==1.005
diag=pd.read_csv(R/'evidence/isolet_query_diagnostics.csv').drop_duplicates('query_id');assert len(diag)==80
assert round(diag.old_largest_layer_fraction.median()*100,1)==99.7
assert round(diag.candidate_kernel_mass_recall.mean()*100,1)==97.2
ss=pd.read_csv(R/'remote/seed_sensitivity/summary.csv');assert len(ss)==63 and (ss.matched_gain_ci_low>0).all()
assert (round(ss.matched_gain_percent.min(),1),round(ss.matched_gain_percent.max(),1))==(13.6,28.9)
report['checked_quantitative_claims']=True
(Q/'PDF_CHECK.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
