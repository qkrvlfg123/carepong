# -*- coding: utf-8 -*-
"""
paired_diff.py — 사람 단위 부트스트랩 쌍대조: 같은 표본에서 AUC(4피처) − AUC(3피처)
=====================================================================
LOSO 예측은 고정, 22명을 복원추출(뽑힌 사람의 창은 통째로) 1000회 · seed 0.
사용:  python paired_diff.py [창초=60]   (maus_our_<W>s.csv 필요)
"""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from maus_feature_eval import pz, mk
W = int(sys.argv[1]) if len(sys.argv) > 1 else 60
df = pd.read_csv(f"maus_our_{W}s.csv").dropna()
y = df["label"].values.astype(int); g = df["subject"].values
def oof_of(feats):
    Xz = pz(df[feats].values.astype(float), g); o = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(Xz, y, g):
        o[te] = mk().fit(Xz[tr], y[tr]).predict_proba(Xz[te])[:, 1]
    return o
o4 = oof_of(["mean_hr", "rmssd", "sdnn", "pnn50"]); o3 = oof_of(["rmssd", "sdnn", "pnn50"])
a4, a3 = roc_auc_score(y, o4), roc_auc_score(y, o3)
subs = np.unique(g); idx = {s: np.where(g == s)[0] for s in subs}
rng = np.random.default_rng(0); d = []; skip = 0
for _ in range(1000):
    i = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
    if len(np.unique(y[i])) < 2: skip += 1; continue
    d.append(roc_auc_score(y[i], o4[i]) - roc_auc_score(y[i], o3[i]))
d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
print(f"{W}초 · {len(df)}창 · {len(subs)}명 · 유효 표본 {len(d)}/1000 (건너뜀 {skip})")
print(f"전체 AUC: 4피처 {a4:.4f}  3피처 {a3:.4f}  차이 {a4-a3:+.4f}")
print(f"부트스트랩 차이: 평균 {d.mean():+.4f}  95% CI [{lo:+.4f} ~ {hi:+.4f}]  차이>0 비율 {(d>0).mean():.3f}")
print(f"(참고) 사람별 단측 p ≈ {(d<=0).mean():.3f}")
