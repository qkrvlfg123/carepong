# -*- coding: utf-8 -*-
"""
maus_build_v5.py — v5 서빙 모델 번들(ppg_stress_base_maus_60s.pkl) 만들기
=====================================================================
maus_our_60s.csv (60초 창 · rest 포함=저부하) 로
  1) LOSO out-of-fold 확률 → loso_auc, Youden 컷(cutoff_prob / cutoff_score)
  2) 전체 데이터로 최종 모델 학습 (개인 z-score → StandardScaler → 로지스틱 balanced)
  3) pop_stats = 저부하(label 0) 창의 평균·표준편차(ddof=0) — 개인 기준선이 없을 때 쓰는 집단 기준
을 묶어 joblib.dump. 참조 pkl이 있으면 값을 비교해 재현 여부를 출력.

사용:  python maus_build_v5.py [출력=ppg_stress_base_maus_60s.pkl] [참조=../ppg_stress_base_maus_60s.pkl]
"""
import os, sys, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve
from maus_feature_eval import pz, mk

FEATS = ["mean_hr", "rmssd", "sdnn", "pnn50"]


def build():
    df = pd.read_csv("maus_our_60s.csv").dropna()
    X = df[FEATS].values.astype(float); y = df["label"].values.astype(int); g = df["subject"].values
    Xz = pz(X, g)
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(Xz, y, g):
        oof[te] = mk().fit(Xz[tr], y[tr]).predict_proba(Xz[te])[:, 1]
    fpr, tpr, th = roc_curve(y, oof); k = np.argmax(tpr - fpr)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xz, y)
    low = df[df["label"] == 0]
    return {
        "model": model,
        "features": FEATS,
        "mode": "personal",
        "pop_stats": {f: (float(low[f].mean()), float(low[f].std(ddof=0))) for f in FEATS},
        "cutoff_prob": float(th[k]),
        "cutoff_score": int(round(th[k] * 100)),
        "min_personal_n": 5,
        "model_version": "base-v5-maus-hr-60s",
        "loso_auc": float(roc_auc_score(y, oof)),
        "note": "4피처(HR추가)·60초 측정. MAUS 재처리. 자체수집 후 컷 재확정.",
    }


def compare(b, r):
    ok = True
    for k in r:
        if k == "model":
            d = max(np.abs(b[k][-1].coef_ - r[k][-1].coef_).max(), np.abs(b[k][-1].intercept_ - r[k][-1].intercept_).max())
            same = d < 1e-9; print(f"  {'OK ' if same else 'DIFF'} model 계수 최대차 {d:.1e}")
        elif k == "pop_stats":
            d = max(abs(a - c) for f in r[k] for a, c in zip(b[k][f], r[k][f]))
            same = d < 1e-9; print(f"  {'OK ' if same else 'DIFF'} pop_stats 최대차 {d:.1e}")
        elif isinstance(r[k], float):
            same = abs(b[k] - r[k]) < 1e-12; print(f"  {'OK ' if same else 'DIFF'} {k}: {b[k]!r} vs {r[k]!r}")
        else:
            same = b.get(k) == r[k]; print(f"  {'OK ' if same else 'DIFF'} {k}: {b.get(k)!r} vs {r[k]!r}")
        ok &= bool(same)
    print("→ 재현 일치" if ok else "→ 차이 있음")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ppg_stress_base_maus_60s.pkl"
    ref = sys.argv[2] if len(sys.argv) > 2 else "../ppg_stress_base_maus_60s.pkl"
    b = build()
    joblib.dump(b, out)
    print(f"저장: {out}  (loso_auc {b['loso_auc']:.4f}, cutoff {b['cutoff_score']})")
    if os.path.exists(ref) and os.path.abspath(ref) != os.path.abspath(out):
        print(f"참조 비교: {ref}"); compare(b, joblib.load(ref))
