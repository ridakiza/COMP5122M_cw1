"""Deeper robustness analysis + independent check of reports/robustness_table.csv.

Run from the repo root (after robustness_check.py):
    python scripts/deep_robustness_check.py

1. Recomputes every gap a second, independent way (cleaned file only) and compares with the table.
2. Compares each flagged change with the 95% margin of error of the gap (Kish weights-only;
   real margins are somewhat larger because Pew's clustering is not included).
3. Which gaps are clearly above/below zero.
4. Why weights move Greece (education mix of the 65+ sample).
5. Turkey's Don't know by age.
6. Bootstrap: how often the largest/smallest country stays the same (takes ~1 minute).
"""
import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
f=pd.read_csv(ROOT/'data/clean/pew_clean_final.csv',low_memory=False,encoding='utf-8-sig')
tab=pd.read_csv(ROOT/'reports/robustness_table.csv')
names={2:'Canada',3:'France',4:'Germany',5:'Greece',6:'Hungary',7:'Italy',8:'Netherlands',9:'Poland',10:'Spain',11:'Sweden',12:'UK',13:'Australia',14:'India',15:'Indonesia',16:'Japan',17:'South Korea',18:'Israel',19:'Turkey',20:'Kenya',21:'Nigeria',22:'South Africa',23:'Argentina',24:'Brazil',25:'Mexico'}
f['cn']=f.country.map(names); f['band']=f.age_band.astype(str).str.replace('–','-')
# independent re-implementation: cleaned file only; blank = DK/Refused
M={'Q30 heard "a lot"':('ai_heard',1),'Q31 "more excited"':('ai_cncexc',1),'Q31 "more concerned"':('ai_cncexc',2)}
def gap(col,code,w=True,dk=True):
    d=f[f.band.isin(['18-29','65+'])].copy()
    if not dk: d=d[d[col].notna()]
    d['y']=(d[col]==code).astype(float); d['wt']=d.weight if w else 1.0
    p=d.groupby(['cn','band']).apply(lambda g: 100*np.average(g.y,weights=g.wt),include_groups=False).unstack()
    return p['18-29']-p['65+']
mx=0
for m,(col,code) in M.items():
    base=gap(col,code)
    for alt,kw in (('weights off',dict(w=False)),("Don't know left out",dict(dk=False))):
        new=gap(col,code,**kw); t=tab[(tab.measure==m)&(tab.alternative==alt)].set_index('country')
        mx=max(mx,(t.gap_main-base.round(1)).abs().max(),(t.gap_alt-new.round(1)).abs().max())
print('1. independent recompute vs robustness_table: max difference',mx)
# 2. are changes larger than sampling noise? SE of weighted gap (Kish eff n)
def se_gap(col,code,dk=True):
    out={}
    for c,g in f[f.band.isin(['18-29','65+'])].groupby('cn'):
        v=0
        for b in ['18-29','65+']:
            h=g[g.band==b]
            if not dk: h=h[h[col].notna()]
            w=h.weight; p=np.average(h[col]==code,weights=w); n=w.sum()**2/(w**2).sum()
            v+=p*(1-p)/n
        out[c]=100*np.sqrt(v)
    return pd.Series(out)
print('\n2. change vs 95% margin of error of the main gap (flagged rows):')
for m,(col,code) in M.items():
    se=se_gap(col,code)
    t=tab[(tab.measure==m)&tab.flag.notna()]
    for _,r in t.iterrows():
        moe=1.96*se[r.country]
        print(f'   {m:22s} {r.alternative:20s} {r.country:12s} change {r.change:+5.1f} | gap {r.gap_main:+5.1f} ±{moe:4.1f} | {"inside" if abs(r.change)<moe else "OUTSIDE"} margin | flag {r.flag.strip()}')
# 3. which gaps are distinguishable from zero / each other
print('\n3. main gaps with 95% margin:')
for m,(col,code) in M.items():
    g=gap(col,code); se=se_gap(col,code); lo=g-1.96*se; hi=g+1.96*se
    pos=(lo>0).sum(); neg=(hi<0).sum(); amb=len(g)-pos-neg
    print(f'   {m}: clearly positive {pos}, clearly negative {neg}, not distinguishable from 0 {amb}: {list(g.index[(lo<=0)&(hi>=0)])}')
    s=g.sort_values(ascending=False); 
    print(f'      top2 {s.index[0]} {s.iloc[0]:.1f}±{1.96*se[s.index[0]]:.1f}, {s.index[1]} {s.iloc[1]:.1f}±{1.96*se[s.index[1]]:.1f} | bottom {s.index[-1]} {s.iloc[-1]:.1f}±{1.96*se[s.index[-1]]:.1f}, {s.index[-2]} {s.iloc[-2]:.1f}±{1.96*se[s.index[-2]]:.1f}')
# 4. why Greece moves with weights: composition of 65+
g=f[(f.cn=='Greece')&(f.band=='65+')]
print(f'\n4. Greece 65+: n={len(g)}; "a lot" unweighted {100*(g.ai_heard==1).mean():.1f}% vs weighted {100*np.average(g.ai_heard==1,weights=g.weight):.1f}%')
for lvl in ['Low','Medium','High']:
    h=g[g.edu_level==lvl]; print(f'   {lvl:6s} share of sample {100*len(h)/len(g):4.1f}% | share after weighting {100*h.weight.sum()/g.weight.sum():4.1f}% | "a lot" {100*(h.ai_heard==1).mean():4.1f}%')
# 5. Turkey DK by age
t=f[f.cn=='Turkey']
print('\n5. Turkey Q31 Don\'t know/Refused (weighted) by age:',{b:round(100*np.average(t[t.band==b].ai_cncexc.isna(),weights=t[t.band==b].weight),1) for b in ['18-29','30-49','50-64','65+']}, '| 65+ n =',int((t.band=='65+').sum()))
# 6. rank stability by bootstrap (Q30 and Q31 excited)
rng=np.random.default_rng(1)
print('\n6. bootstrap (500 resamples within country/age): how often the extremes stay')
for m in ['Q30 heard "a lot"','Q31 "more excited"']:
    col,code=M[m]; base=gap(col,code); top=base.idxmax(); bot=base.idxmin(); kt=kb=0
    d=f[f.band.isin(['18-29','65+'])][['cn','band','weight',col]]
    grp={k:v for k,v in d.groupby(['cn','band'])}
    for _ in range(500):
        res={}
        for c in base.index:
            v=[]
            for b in ['18-29','65+']:
                h=grp[(c,b)]; i=rng.integers(0,len(h),len(h)); hh=h.iloc[i]
                v.append(100*np.average(hh[col]==code,weights=hh.weight))
            res[c]=v[0]-v[1]
        s=pd.Series(res); kt+=s.idxmax()==top; kb+=s.idxmin()==bot
    print(f'   {m}: {top} largest in {kt/5:.0f}% | {bot} smallest in {kb/5:.0f}%')