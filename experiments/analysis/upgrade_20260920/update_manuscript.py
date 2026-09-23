from pathlib import Path
import shutil,difflib,re,json
root=Path(__file__).resolve().parents[1]
base=root/'baseline_backup/final_verification_20260920/manuscript'
out=root/'manuscript'
shutil.copytree(base,out)
oldname='ICASSP2027_EN_revised_final.tex'
s=(base/oldname).read_text(); original=s
changes=[]
def replace(a,b,why):
 global s
 assert s.count(a)==1,(why,s.count(a))
 s=s.replace(a,b);changes.append(why)
replace('The gain weakens for narrow bandwidths and depends on budget at larger reference sizes, while complete-query advantages remain restricted after stronger DEANN tuning.',r'The gain weakens for narrow bandwidths; equivalent multirank partitioning reduces complete-query latency by 40.4--42.9\% at 20,000 GIST references, while smaller-set computational advantages remain restricted.','Abstract: bounded verified implementation gain')
replace(r'\textit{Third}, bandwidth, reference-size and complete-query measurements delimit its operating range against validation-tuned DEANN.',r'\textit{Third}, complete-query measurements quantify a scale-dependent gain from equivalent partition construction and delimit the operating range against validation-tuned DEANN.','Contribution: measured implementation value without claiming new selection theory')
replace('H$ need not contain the true nearest neighbors.','H$ need not contain the true nearest neighbors. An equivalent multirank implementation selects cell boundaries, falling back to the original sort for boundary ties; memberships and sampling laws remain unchanged.','Method: exact boundary selection and tie fallback')
replace('Online work is $O(dr+nr+n\\log n+Md)$:', 'Full-sort online work is $O(dr+nr+n\\log n+Md)$:', 'Complexity: retain full-scan and full-sort scope')
replace('These particular coarse cells motivate the variance diagnosis; they are not a claim about every norm--angle configuration.','', 'Remove repeated diagnostic qualifier; first sentence and exact configuration retained')
replace(r'Official DEANN~\cite{deann} uses permuted sampling with the FAISS IVF index~\cite{faiss}. Stronger validation tests 24 $(n_{\rm probe},k/M)$ pairs on all 80 validation queries at $M=64,128,256$ and two sampling seeds. Dataset-wide choices minimizing error, latency and their product are frozen before test/audit, yielding two distinct configurations per dataset; Fig.~\ref{fig:latency} also retains the original. The IVF list count stays fixed. Actual kernel calls include overlap corrections.',r'Official DEANN~\cite{deann} uses permuted sampling with FAISS IVF~\cite{faiss}. A 24-point $(n_{\rm probe},k/M)$ grid on 80 validation queries at $M=64,128,256$ and two seeds freezes dataset-wide error- and latency-selected curves before test/audit. IVF list counts stay fixed; kernel counts include overlap corrections. Fig.~\ref{fig:latency} and Table~\ref{tab:working} retain the original full-sort projection implementation.','Compress unchanged DEANN protocol and distinguish frozen figures from new cost study')
replace(r'At $n=20{,}000$, sorting plus strata construction takes 78\% of mean $M=128$ query time. Statistical recovery therefore depends on budget, while the full low-dimensional scan and sorting limit computational utility.',r'A frozen interleaved comparison on these queries attributes 74.9\% of instrumented mean time to sorting alone. Equivalent multirank construction reduces median complete-query time by 40.4--42.9\% at $M=64/128/256$ (1.801 to 1.036 ms at 128; paired intervals exclude zero). At $n=720/1800$, changes range from 0.3\% faster to 4.4\% slower. The implementation gain is scale-specific; statistical recovery still depends on budget.','RQ4: formal complete-query values, positive intervals and small-n regressions')
replace('Stronger DEANN leaves selected frontier points, delimiting the computational value of the statistical improvement.',r'Equivalent multirank construction cuts query time by 40.4--42.9\% at 20,000 GIST references, with no corresponding small-set gain. Stronger DEANN still limits the computational operating range.','Conclusion: preserve computational boundary')
replace('An equivalent multirank implementation selects cell boundaries, falling back to the original sort for boundary ties; memberships and sampling laws remain unchanged.','Exact multirank selection with full-sort fallback at boundary ties preserves the same $H$, cells and sampling laws.','Length and scope clarification; unchanged scientific content')
replace('This comparator is separate from the final estimator. It combines floor-based norm bins','A separate diagnostic combines floor-based norm bins','Length and scope clarification; unchanged scientific content')
replace('Additional interventions use the existing 80 validation queries per dataset, $M=128$ and fixed $H$ of size 32.','Additional interventions use 80 validation queries per dataset, $M=128$ and $|H|=32$.','Length and scope clarification; unchanged scientific content')
replace('attributes 74.9\\% of instrumented mean time to sorting alone.','attributes 74.9\\% of $M=128$ instrumented mean time to sorting alone.','Length and scope clarification; unchanged scientific content')
replace('Projected ordering retains substantial exact-stratification gain in the evaluated Gaussian KDE settings.','Projected ordering retains substantial exact-stratification gain for the evaluated Gaussian KDE targets.','Length and scope clarification; unchanged scientific content')
replace('Partition interventions connect the increment to conditional variance and expose population-imbalance failures. Narrow bandwidths weaken the gain; larger reference sets require adequate budgets.','Partition interventions connect this gain to conditional variance and expose population imbalance. Narrow kernels weaken it; larger sets require adequate budgets.','Length and scope clarification; unchanged scientific content')
replace('with no corresponding small-set gain. Stronger DEANN still limits the computational operating range.','with no small-set gain. Stronger DEANN limits the computational operating range.','Length and scope clarification; unchanged scientific content')
replace('OpenAI Codex assisted revision of the author-supplied manuscript and the development and checking of experiment code and figures.','OpenAI Codex assisted revision of the supplied manuscript and development and verification of experiment code and figures.','Length and scope clarification; unchanged scientific content')
a=s.index('\\subsection{RQ4:'); b=s.index('\\section{Conclusion}',a)
s=s[:a]+(root/'scripts/rq4_compact.tex').read_text()+s[b:]
changes.append('Compress RQ4 prose while retaining all quantitative facts and costs')
name='ICASSP2027_EN_revised_upgrade.tex'
(out/name).write_text(s)
for duplicate in out.glob('ICASSP2027_EN_revised_final.*'):
    duplicate.unlink()
(root/'MANUSCRIPT.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='frozen/'+oldname,tofile='upgrade/'+name)))
a=s.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
(root/'MANUSCRIPT_CHANGE_RECORD.json').write_text(json.dumps(dict(changes=changes,abstract_words=len(a.split()),original_file=oldname,new_file=name,baseline_tables_figures_unchanged=True),indent=2))
print('Abstract words:',len(a.split()))
