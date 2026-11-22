# -*- coding: utf-8 -*-
"""
maus_window_eval.py — 창 길이별(30/60/120초) 모델 성능 비교
=========================================================
목적: 우리 파이프라인으로 재처리한 MAUS 데이터에서, 창 길이를 줄여도
      스트레스(과부하) 판정 성능이 유지되는지 사람 단위 교차검증(LOSO)으로 비교.
      → '30초 측정도 쓸 만하다'의 정량적 근거.

전처리·모델은 배포 모델과 동일:
  개인(피험자) 기준선 z-score → 로지스틱 회귀 → LOSO AUC.

사용:  python maus_window_eval.py
  (같은 폴더에 maus_our_30s.csv / _60s.csv / _120s.csv 필요)
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

FEATS = ["rmssd", "sdnn", "pnn50"]


def personal_z(X, g):
    Xz = X.copy().astype(float)
    for s in np.unique(g):
        m = g == s
        mu, sd = X[m].mean(0), X[m].std(0); sd[sd == 0] = 1.0
        Xz[m] = (X[m] - mu) / sd
    return Xz


def mk():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


def eval_window(W):
    fn = f"maus_our_{W}s.csv"
    try:
        df = pd.read_csv(fn)
    except FileNotFoundError:
        print(f"  [{W}초] {fn} 없음 — 건너뜀"); return None
    if len(df) < 20:
        print(f"  [{W}초] 표본 부족({len(df)})"); return None
    X = df[FEATS].values.astype(float)
    y = df["label"].values.astype(int)
    g = df["subject"].values
    Xz = personal_z(X, g)
    logo = LeaveOneGroupOut(); oof = np.zeros(len(y))
    for tr, te in logo.split(Xz, y, g):
        m = mk(); m.fit(Xz[tr], y[tr]); oof[te] = m.predict_proba(Xz[te])[:, 1]
    auc = roc_auc_score(y, oof)
    # 부트스트랩 CI
    rng = np.random.default_rng(0); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], oof[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"W": W, "n": len(y), "subj": df["subject"].nunique(),
            "auc": auc, "lo": lo, "hi": hi}


if __name__ == "__main__":
    print("=" * 56)
    print(" 창 길이별 성능 (우리 파이프라인 재처리 · LOSO)")
    print("=" * 56)
    res = [eval_window(W) for W in (30, 60, 120)]
    res = [r for r in res if r]
    print(f"\n{'창':>6} {'표본':>6} {'피험자':>6} {'LOSO AUC':>10} {'95% CI':>18}")
    for r in res:
        print(f"{r['W']:>5}초 {r['n']:>6} {r['subj']:>6} {r['auc']:>10.3f}   [{r['lo']:.3f}~{r['hi']:.3f}]")
    if res:
        base = next((r for r in res if r["W"] == 120), res[-1])
        print("\n해석:")
        for r in res:
            d = r["auc"] - base["auc"]
            print(f"  {r['W']}초는 120초 대비 {d:+.3f}")
        print("\n→ 창을 줄여도 AUC가 크게 떨어지지 않으면 '30초 측정'이 정당화된다.")
