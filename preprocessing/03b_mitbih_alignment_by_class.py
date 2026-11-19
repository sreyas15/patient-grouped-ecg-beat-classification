"""R2 Stage 1c-bis: MIT-BIH R-peak-proxy alignment by AAMI class (comparator for the INCART audit; same measure as 03_audit_incart.py)."""
import os, json, numpy as np, wfdb
HERE=os.path.dirname(os.path.abspath(__file__)); R2=os.path.dirname(HERE)
MIT=r"D:\explainable_ai\Journel paper project\WORKFLOW\Dataset\mit-bih-arrhythmia-database-1.0.0"
CS=['N','L','R','e','j','A','a','J','S','V','E','F','/','f','Q']
def aami(s): i=CS.index(s); return 'N' if i<=4 else 'S' if i<=8 else 'V' if i<=10 else 'F' if i==11 else 'Q'
h=27; off={c:[] for c in 'NSVFQ'}
for r in [l.strip() for l in open(os.path.join(MIT,'RECORDS')) if l.strip()]:
    x=wfdb.rdrecord(os.path.join(MIT,r),channels=[0]).p_signal[:,0]; a=wfdb.rdann(os.path.join(MIT,r),'atr')
    for t,s in zip(a.sample,a.symbol):
        if s in CS and h<=t<len(x)-h:
            w=x[t-h:t+h+1]; off[aami(s)].append((np.argmax(np.abs(w-np.median(w)))-h)*1000/360)
q=lambda v:dict(p05=round(float(np.percentile(v,5)),2),median=round(float(np.median(v)),2),p95=round(float(np.percentile(v,95)),2),frac_abs_le_14ms=round(float((np.abs(v)<=14).mean()),3),n=len(v))
res={c:q(np.array(v)) for c,v in off.items()}
json.dump(res,open(os.path.join(R2,'results','data_audit','mitbih_alignment_by_class.json'),'w'),indent=1); print(json.dumps(res,indent=1))
