# -*- coding: utf-8 -*-
"""
maus_feature_eval.py — 피처 조합 비교 (30초 창)
==============================================
목적: 우리 파이프라인 재처리 데이터로 '어떤 HRV 피처 조합이 최선인지' 비교.
      특히 mean_hr(평균 심박)을 넣으면 성능이 오르는지 확인 → 피처 선택 근거.
전처리·모델은 배포와 동일: 개인 기준선 z-score → 로지스틱 → LOSO AUC.
사용:  python maus_feature_eval.py [창초=30]   (maus_our_<W>s.csv 필요)
"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

def pz(X, g):
    Xz = X.copy().astype(float)
    for s in np.unique(g):
        m = g == s; mu, sd = X[m].mean(0), X[m].std(0); sd[sd == 0] = 1.0; Xz[m] = (X[m]-mu)/sd
    return Xz

def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))

def auc_of(df, feats):
    X = df[feats].values.astype(float); y = df["label"].values.astype(int); g = df["subject"].values
    Xz = pz(X, g); logo = LeaveOneGroupOut(); oof = np.zeros(len(y))
    for tr, te in logo.split(Xz, y, g):
        m = mk(); m.fit(Xz[tr], y[tr]); oof[te] = m.predict_proba(Xz[te])[:, 1]
    auc = roc_auc_score(y, oof)
    rng = np.random.default_rng(0)
    bs = [roc_auc_score(y[i], oof[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if len(np.unique(y[i])) == 2]
    # 사람 단위 부트스트랩: 피험자를 복원추출, 뽑힌 사람의 창은 통째로 포함
    subs = np.unique(g); idx = {s: np.where(g == s)[0] for s in subs}; rng = np.random.default_rng(0); bs_s = []
    for _ in range(1000):
        i = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(np.unique(y[i])) == 2:
            bs_s.append(roc_auc_score(y[i], oof[i]))
    return auc, np.percentile(bs, 2.5), np.percentile(bs, 97.5), np.percentile(bs_s, 2.5), np.percentile(bs_s, 97.5)

def main(W=30):
    df = pd.read_csv(f"maus_our_{W}s.csv").dropna()
    combos = {
        "① RMSSD·SDNN·pNN50 (현재 배포)": ["rmssd", "sdnn", "pnn50"],
        "② + 평균심박(HR)": ["mean_hr", "rmssd", "sdnn", "pnn50"],
        "③ + HR·SD1·SD2 (전체)": ["mean_hr", "rmssd", "sdnn", "pnn50", "sd1", "sd2"],
        "④ HR 단독": ["mean_hr"],
        "⑤ HR + RMSSD": ["mean_hr", "rmssd"],
    }
    print("=" * 60)
    print(f" 피처 조합별 성능 ({W}초 창 · LOSO · {len(df)}창 · {df.subject.nunique()}명)")
    print("=" * 60)
    print(f"{'조합':32} {'AUC':>7}   {'95% CI (창 단위)':16} {'95% CI (사람 단위)'}")
    best = None
    for name, feats in combos.items():
        a, lo, hi, slo, shi = auc_of(df, feats)
        print(f"{name:32} {a:>7.3f}   [{lo:.3f}~{hi:.3f}]    [{slo:.3f}~{shi:.3f}]")
        if best is None or a > best[1]:
            best = (name, a)
    print("-" * 60)
    print(f"최고: {best[0]} (AUC {best[1]:.3f})")
    print("\n해석: ②·③이 ①보다 높으면 평균심박(HR) 추가가 유효 → 배포 모델에 HR 추가 검토.")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
