"""Build every table of the manuscript from the saved K5A reports (ERNP = --n_prior 5 --keep A). No model is run.
usage (from the repository root): python tables_figures/build_tables.py   -> tables_figures/tables/*.tex
ERNP_PAPER_DIR=<manuscript folder> writes the tables to <folder>/tables/ and keeps only those the manuscript inputs."""
from pathlib import Path
import os, json, shutil, hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = Path(os.environ.get('ERNP_PAPER_DIR', HERE)).resolve()
OUT = PAPER / 'tables'
OUT.mkdir(exist_ok=True)
SECONDARY = ROOT/'exp/reports'   # every report of the final configuration (K5A)
# Reuse the existing secondary-table transformations in an isolated output folder.
LEGACY=HERE/'generator/secondary_tables.py'
if not LEGACY.exists(): LEGACY=ROOT/'paper/make_tables.py'
code=LEGACY.read_text(encoding='utf-8')
rep_line=next(line for line in code.splitlines() if line.startswith('REP ='))
code=code.replace(rep_line, f'REP = {str(SECONDARY)!r}; OUT = {str(OUT)!r}; os.makedirs(OUT, exist_ok=True)')
exec(compile(code, str(LEGACY), 'exec'), {'__file__': str(LEGACY)})

AUDIT=SECONDARY   # exp/access_report.py, exp/ablation_report.py
runs = pd.read_csv(AUDIT / 'main_runs.csv')
means = pd.read_csv(AUDIT / 'dataset_means.csv')
cohorts = pd.read_csv(AUDIT / 'cohort_means.csv')
tests = pd.read_csv(AUDIT / 'dataset_tests.csv')
DS = ['CIDDS_ctx', 'ToN_ctx', 'CIC17', 'CIC18', 'CTU13', 'SIMARGL', 'APPRAISE']
DSN = dict(zip(DS, ['CIDDS', 'ToN', 'CIC17', 'CIC18', 'CTU-13', 'SIMARGL (filtered)', 'APPRAISE']))
OURS = 'ernp_K5A'
MS = [OURS, 'ssf_causal', 'cade', 'trident', 'memstream', 'owad', 'knn']
MN = dict(zip(MS, [r'\textbf{ERNP}', 'SSF', 'CADE', 'Trident', 'MemStream', 'OWAD', r'$k$NN']))
def table(name, spec, header, rows):
    (OUT / name).write_text('\\begin{tabular}{'+spec+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
def f(v): return f'{v:.3f}'
hist = runs[runs.dataset.isin(DS)]
for metric, name in [('F1', 'main_f1.tex'), ('MCC', 'main_mcc.tex'), ('AUROC', 'main_auc.tex'), ('zd_recall_at1fpr', 'main_zd.tex')]:
    g = hist.groupby(['dataset','method'])[metric].agg(['mean','std'])
    rows=[]
    for m in MS:
        cells=[]
        for ds in DS:
            mu, sd = g.loc[(ds,m)]
            cell=f'{mu:.3f}$\\pm${sd:.3f}'
            if np.isclose(mu, g.loc[ds]['mean'].max()): cell=r'\textbf{'+cell+'}'
            cells.append(cell)
        cells.append(f(g.xs(m,level='method')['mean'].mean()))
        rows.append(MN[m]+' & '+' & '.join(cells)+r'\\')
    table(name,'l'+'c'*8,'Method & CIDDS & ToN & CIC17 & CIC18 & CTU-13 & SIMARGL$^*$ & APPRAISE & Mean',rows)

cn = {'historical_filtered_7':'Filtered (7)', 'published_labels_7':'Published labels (7)', 'without_uncertain_SIMARGL_6':'Without SIMARGL (6)'}
rows=[]
for c,label in cn.items():
    part=cohorts[cohorts.cohort.eq(c)].set_index('method')
    for m in MS:
        z=part.loc[m]
        rows.append((label if m==MS[0] else '')+' & '+MN[m]+' & '+' & '.join(f(z[k]) for k in ['F1','MCC','AUROC','zd_recall_at1fpr','macro_zd_first_period_at1fpr'])+r'\\')
    if c!=list(cn)[-1]: rows.append(r'\midrule')
table('cohorts.tex','llccccc',r'Label policy & Method & F1 & MCC & AUROC & Pooled ZD@1\% & First-period macro@1\%', rows)
rows=[]
for ds in DS+['SIMARGL_all']:
    z=means[means.dataset.eq(ds)&means.method.eq(OURS)].iloc[0]
    label='SIMARGL (published)' if ds=='SIMARGL_all' else DSN[ds]
    rows.append(label+' & '+' & '.join(f(z[k]) for k in ['F1','MCC','AUROC'])+' & '+f'{100*z.FPR:.2f}'+ ' & '+' & '.join(f(z[k]) for k in ['zero_day_recall','zd_recall_at1fpr','macro_zd_first_period_recall','macro_zd_first_period_at1fpr'])+r'\\')
table('ernp_detail.tex','lcccccccc',r'Dataset & F1 & MCC & AUROC & FPR (\%) & ZD own & ZD@1\% & First own & First@1\%',rows)
# paired tests, one row per detector (wins and two-sided p are printed in fig_paired, the main-text figure)
TM=['F1','MCC','AUROC','zd_recall_at1fpr']; tz=tests[tests.cohort.eq('historical_filtered_7')].set_index(['metric','baseline'])
assert np.allclose(2*tz.p_one_sided, tz.p_two_sided)   # every difference is in ERNP's favour on balance: p1 = p2 / 2
rows=[MN[m]+' & '+' & '.join(f'{tz.loc[(k,m)].mean_diff:+.3f} & {tz.loc[(k,m)].p_one_sided_holm:.4f} & {tz.loc[(k,m)].p_two_sided_holm:.4f}' for k in TM)+r'\\' for m in MS[1:-1]]
table('statistics.tex','l'+'ccc'*4,' & '+' & '.join(r'\multicolumn{3}{c}{'+h+'}' for h in ['F1','MCC','AUROC',r'ZD@1\%'])+r'\\'+'\n'+
      ''.join(rf'\cmidrule(lr){{{2+3*i}-{4+3*i}}}' for i in range(4))+'\n'+
      'Detector & '+' & '.join([r'$\Delta$ & Holm$_5$($p_1$) & Holm$_5$($p_2$)']*4),rows)
# Paired resampling intervals describe sample variation, not independent deployments.
rng=np.random.default_rng(0); rows=[]
for ds in DS:
    p=hist[hist.dataset.eq(ds)].pivot(index='seed',columns='method',values='F1')
    cells=[]
    for m in MS[1:-1]:
        delta=(p[OURS]-p[m]).dropna().to_numpy()
        lo,hi=np.quantile(rng.choice(delta,(10000,len(delta))).mean(axis=1),[.025,.975])
        cells.append(f'{delta.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')
    rows.append(DSN[ds]+' & '+' & '.join(cells)+r'\\')
table('effects.tex','lccccc','Dataset & SSF & CADE & Trident & MemStream & OWAD',rows)
bench=json.loads((AUDIT/'benchmark_k5.json').read_text())
table('benchmark.tex','lcccc','Dataset & Models & Initial fit (s) & Median score (s) & Flows/s',[
    DSN[a['dataset']]+f" & {a['models_trained']} & {a['initial_fit_s']:.2f} & {a['median_score_s']:.2f} & {a['median_flows_per_s']:,.0f}"+r'\\' for a in bench['rows']])
shutil.copy2(HERE/'evidence/datasets.tex',OUT/'datasets.tex')
# CIC17 influenced the final choice of K, so the old held-out dagger is removed.
for p in OUT.glob('*.tex'):
    p.write_text(p.read_text(encoding='utf-8').replace(r'$^\dagger$', ''), encoding='utf-8')
scale = pd.read_csv(SECONDARY/'scale_runs.csv')
g = scale.groupby(['dataset','method'])[['F1','MCC','zd_recall_at1fpr']].agg(['mean','std'])
table('full_scale.tex','lccc',r'Method & F1 & MCC & ZD@1\%',[
    MN[m]+' & '+' & '.join(f"{g.loc[('CIC17_full','ernp' if m==OURS else m),(k,'mean')]:.3f}$\\pm${g.loc[('CIC17_full','ernp' if m==OURS else m),(k,'std')]:.3f}" for k in ['F1','MCC','zd_recall_at1fpr'])+r'\\' for m in MS[:-1]])
noise=pd.read_csv(SECONDARY/'noise_runs.csv')
g=noise.groupby(['dataset','method'])['F1'].mean()
table('noise.tex','lccccc',r'Dataset & ERNP 5\% & ERNP 10\% & ERNP 20\% & SSF 10\% & OWAD 10\%',[
    DSN[ds]+' & '+' & '.join(f(g.loc[ds,m]) for m in ['ernp_N5','ernp_N10','ernp_N20','ssf_causal_N10','owad_N10'])+r'\\' for ds in DS[:4]])
# component ablations of the final configuration on the seven datasets (exp/ablation_report.py)
abl=pd.read_csv(AUDIT/'ablation_runs.csv'); af=abl.groupby(['dataset','method'])[['F1','MCC']].mean()
at=pd.read_csv(AUDIT/'ablation_tests.csv'); at=at[at.metric=='F1'].set_index('ablation')
AB=[('ernp_B0_K5A','No online labels'),('abl_evid_rep_K5A','Representative-only labels'),('abl_retr_none_K5A','No retraction'),
    ('abl_prior_scarf_K5A','Independent-donor (SCARF) prior'),('abl_prior_none_K5A','No novelty prior (GBM+AL)')]
rows=[]
for m,lab in AB:
    d=[af.loc[(ds,m),'F1']-af.loc[(ds,OURS),'F1'] for ds in DS]
    dm=np.mean([af.loc[(ds,m),'MCC']-af.loc[(ds,OURS),'MCC'] for ds in DS])
    ph=at.loc[m,'p_one_sided_holm']
    rows.append(lab+' & '+' & '.join(f'{(0.0 if abs(v)<5e-4 else v):+.3f}' for v in d)+f' & {np.mean(d):+.3f} & {dm:+.3f} & {sum(round(v,3)<0 for v in d)}/7 & {ph:.3f}'+r'\\')
rows.insert(0,r'\ERNP (F1) & '+' & '.join(f(af.loc[(ds,OURS),'F1']) for ds in DS)+f" & {np.mean([af.loc[(ds,OURS),'F1'] for ds in DS]):.3f} & & & "+r'\\'); rows.insert(1,r'\midrule')
table('ablation.tex','l'+'c'*11,r'Variant ($\Delta$ vs.\ ERNP) & CIDDS & ToN & CIC17 & CIC18 & CTU-13 & SIMARGL$^*$ & APPRAISE & Mean $\Delta$F1 & Mean $\Delta$MCC & Worse & $p_{\mathrm{Holm}}$',rows)
manifest={}
# footprint of wrongly confirmed benign groups (exp/footprint_report.py), delta = 0.5
FP=ROOT/'exp/reports/footprint_runs.csv'
fp=pd.read_csv(FP); fp=fp[fp.delta.eq(0.5)]
assert fp.groupby('dataset').size().eq(5).all() and set(fp.dataset)==set(DS), 'footprint table needs 5 seeds of every dataset'
FN=dict(zip(DS,['CIDDS-001','NF-ToN-IoT-v3','CIC-IDS2017','CSE-CIC-IDS2018','CTU-13','SIMARGL2021$^*$','APPRAISE']))
def pooled(g,col,n):
    w=g[n].sum(); return '--' if w<10 else f'{(g[col].fillna(0)*g[n]).sum()/w:.3f}'
rows=[]
for ds in DS:
    g=fp[fp.dataset.eq(ds)].copy(); g['n_out']=g.n_zd-g.exact_n_zd_in
    rows.append(FN[ds]+f' & {g.n_impure.mean():.1f} & {100*g.bound_share_benign.mean():.1f} & {100*g.bound_share_zd.mean():.1f}'
                f' & {100*g.exact_share_benign.mean():.2f} & {100*g.exact_share_zd.mean():.2f} & {g.exact_n_zd_in.mean():.1f}'
                f' & {pooled(g,"exact_zd_recall_in","exact_n_zd_in")} & {pooled(g,"exact_zd_recall_out","n_out")}'+r'\\')
table('footprint.tex','lcccccccc',r' & & \multicolumn{2}{c}{Bound of Prop.~\ref{prop:p1}(iii), \%} & \multicolumn{3}{c}{Exact region} & \multicolumn{2}{c}{Zero-day recall}\\'
      '\n'+r'\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}'+'\n'+
      r'Dataset & Impure groups & benign & zero-day & benign \% & zero-day \% & zero-day flows & inside & outside',rows)
# RQ2 with an analyst who accepts the ramp as benign (exp/naive_report.py); R0 = correct analyst (same runs as tab:poison)
NV=ROOT/'exp/reports/naive_runs.csv'
nv=pd.read_csv(NV)
NM=[('ERNP',r'\textbf{ERNP}'),('ERNP, rep.-only labels','ERNP, rep.-only labels'),('ERNP, no retraction','ERNP, no retraction'),
    ('GBM+AL (no prior)','GBM+AL (no prior)'),('SSF','SSF')]
NL=[('R0','correct'),('R50','accepts first half'),('R100','accepts whole ramp')]
sel=nv[nv.method.isin([m for m,_ in NM])]
assert sel.groupby(['scenario','method','level']).size().eq(5).all() and len(sel)==2*len(NM)*3*5, 'naive table needs 5 seeds of every cell'
g=sel.groupby(['scenario','method','level']); mu=g[['payload_recall','payload_at1fpr','FPR']].mean(); ab=g['absorbed'].sum()
rows=[]
for i,(m,lab) in enumerate(NM):
    if i: rows.append(r'\midrule')
    for j,(lv,an) in enumerate(NL):
        cells=[]
        for sc in ('init','resp'):
            x=mu.loc[(sc,m,lv)]; cells.append(f'{x.payload_recall:.3f} & {x.payload_at1fpr:.3f} & {100*x.FPR:.1f} & {int(ab[(sc,m,lv)])}/5')
        rows.append((lab if j==0 else '')+f' & {an} & '+' & '.join(cells)+r'\\')
table('naive.tex','ll'+'cccc'*2,r' & & \multicolumn{4}{c}{Compromised initiator} & \multicolumn{4}{c}{Compromised responder}\\'
      '\n'+r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}'+'\n'+
      r'Method & Analyst & payload & payload@1\% & FPR\,\% & absorbed & payload & payload@1\% & FPR\,\% & absorbed',rows)
# analyst load with and without retraction, each with its own feedback (exp/review3_report.py, part 1)
RL=ROOT/'exp/reports/review3_retraction_load.csv'
rl=pd.read_csv(RL)
assert rl.groupby(['dataset','method']).size().eq(5).all() and set(rl.dataset)==set(DS) and len(rl)==2*7*5, 'retraction-load table needs 5 seeds of both runs on all seven datasets'
pv=rl.pivot_table(index=['dataset','seed'],columns='method',values=['benign_alerts_per_day','benign_decisions','F1'])
pct=lambda v: '0' if abs(v)<0.5 else f'{v:+.0f}'
rows=[]
for ds in DS:
    x=pv.loc[ds]; cells=[]
    for v,fm in (('benign_alerts_per_day',',.0f'),('benign_decisions',',.0f')):
        a,b=x[(v,OURS)],x[(v,'abl_retr_none_K5A')]
        cells.append(f'{a.mean():{fm}} & {b.mean():{fm}} & {pct(100*(b.mean()/a.mean()-1))}\\% ({int((b>a).sum())}/5)')
    dF=x[('F1',OURS)]-x[('F1','abl_retr_none_K5A')]
    rows.append(FN[ds]+' & '+' & '.join(cells)+f' & {dF.mean():+.3f}'+r'\\')
table('retload.tex','lccccccc',r' & \multicolumn{3}{c}{Benign alerts per day} & \multicolumn{3}{c}{Benign group decisions per run} & \\'
      '\n'+r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}'+'\n'+
      r'Dataset & \ERNP & no retr. & change (seeds) & \ERNP & no retr. & change (seeds) & $\Delta$F1',rows)
# replays of ERNP K5A's feedback schedule and ToN at the 4 h cadence (exp/review3_report.py, parts 2-3);
# replaces the replay table of the shared transformation, whose no-prior replay followed the single-draw run's schedule
PR=ROOT/'exp/reports/review3_prior.csv'
pr=pd.read_csv(PR)
RPM=[('ernp_K5A',r'\ERNP'),('rpl_retr_none_K5A','no retraction'),('rpl_prior_none_K5A','no prior')]
CDM=[('ernp_P4_K5A',r'\ERNP'),('abl_prior_none_P4_K5A','no prior, own feedback'),('rpl_prior_none_P4_K5A','no prior, \\ERNP\'s schedule')]
RCOL=[('F1','.3f',1),('MCC','.3f',1),('FPR','.1f',100),('AUROC','.3f',1),('zd_recall_at1fpr','.3f',1),('benign_alerts_per_day',',.0f',1)]
rp=pr[pr.block.eq('replay')]; cd=pr[pr.block.eq('cadence')]
assert rp.groupby(['dataset','method']).size().eq(5).all() and len(rp)==4*3*5, 'replay table needs 5 seeds per cell'
assert cd[cd.method.isin([m for m,_ in CDM])].groupby('method').size().eq(5).all(), 'cadence block needs 5 seeds per run'
rm=rp.groupby(['dataset','method'])[[c for c,_,_ in RCOL]].mean(); cm=cd.groupby('method')[[c for c,_,_ in RCOL]].mean()
cells=lambda x: ' & '.join(f'{k*x[c]:{fm}}' for c,fm,k in RCOL)
rows=[]
for i,ds in enumerate(['CIDDS_ctx','ToN_ctx','CIC17','CIC18']):
    if i: rows.append(r'\midrule')
    for j,(m,lab) in enumerate(RPM): rows.append((FN[ds] if j==0 else '')+f' & {lab} & '+cells(rm.loc[(ds,m)])+r'\\')
rows.append(r'\midrule')
for j,(m,lab) in enumerate(CDM): rows.append((r'ToN, 4\,h updates' if j==0 else '')+f' & {lab} & '+cells(cm.loc[m])+r'\\')
table('replay.tex','llcccccc',r'Dataset & Model & F1 & MCC & FPR\,\% & AUROC & ZD@1\% & Benign alerts/day',rows)
# panel (b) of the merged RQ4 mechanism table: the replays above plus the benign FPR by distance to the confirmed benign
# evidence (exp/reports/e3_regions.csv, the same rates as the former retraction table; ernp / rpl_retr_none only)
RR=SECONDARY/'e3_regions.csv'
rr=pd.read_csv(RR); rr=rr[rr.method.isin(['ernp','rpl_retr_none'])]
s=rr.groupby(['dataset','method','bin','seed'])[['fp','n_benign']].sum()
fprb=(100*s.fp/s.n_benign.replace(0,np.nan)).groupby(['dataset','method','bin']).mean()
VIEW={'ernp_K5A':'ernp','rpl_retr_none_K5A':'rpl_retr_none'}
def dist(ds,m):
    if m not in VIEW: return '-- & --'
    return f"{fprb[(ds,VIEW[m],'0-1')]:.2f} & {fprb[(ds,VIEW[m],'3-inf')]:.2f}"
cellsb=lambda x,d: f"{x.F1:.3f} & {x.MCC:.3f} & {100*x.FPR:.1f} & {d} & {x.AUROC:.3f} & {x.zd_recall_at1fpr:.3f} & {x.benign_alerts_per_day:,.0f}"
rows=[]
for i,ds in enumerate(['CIDDS_ctx','ToN_ctx','CIC17','CIC18']):
    if i: rows.append(r'\midrule')
    for j,(m,lab) in enumerate(RPM): rows.append((FN[ds] if j==0 else '')+f' & {lab} & '+cellsb(rm.loc[(ds,m)],dist(ds,m))+r'\\')
rows.append(r'\midrule')
for j,(m,lab) in enumerate(CDM): rows.append((r'ToN, 4\,h updates' if j==0 else '')+f' & {lab} & '+cellsb(cm.loc[m],'-- & --')+r'\\')
table('mech_b.tex','llcccccccc',r' & & & & & \multicolumn{2}{c}{FPR\,\% at distance} & & & \\'
      '\n'+r'\cmidrule(lr){6-7}'+'\n'+
      r'Dataset & Model & F1 & MCC & FPR\,\% & $\le\epsilon$ & $>3\epsilon$ & AUROC & ZD@1\% & Benign alerts/day',rows)
for p in [AUDIT/'main_runs.csv',AUDIT/'ablation_runs.csv',AUDIT/'ablation_tests.csv',AUDIT/'dataset_means.csv',AUDIT/'cohort_means.csv',AUDIT/'dataset_tests.csv',AUDIT/'benchmark_k5.json',FP,NV,RL,PR,RR,ROOT/'exp/reports/ssfg_runs.csv']:
    manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(HERE/'table_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
# The shared transformation also emits tables the manuscript does not use; with a manuscript folder, drop them.
import re
article=''.join(p.read_text(encoding='utf-8') for p in PAPER.glob('*.tex') if '\\documentclass' in p.read_text(encoding='utf-8'))
if article:
    used={name if name.endswith('.tex') else name+'.tex' for name in re.findall(r'\\input\{tables/([^}]+)\}',article)}
    for p in OUT.glob('*.tex'):
        if p.name not in used: p.unlink()
print(f'Tables generated from the saved reports -> {OUT}')
