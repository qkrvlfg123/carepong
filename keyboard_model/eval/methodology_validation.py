# -*- coding: utf-8 -*-
"""
methodology_validation.py - 키보드·마우스 모델의 검증 방법 점검

SWELL-KW 분당 행동 피처로 세 가지를 확인함.
  1) 확률 보정: 학습 / 보정 / 평가를 사람 단위로 분리한 뒤 Isotonic 보정 전후 Brier 비교
  2) Nested CV: 안쪽에서 규제 강도 C 선택, 바깥에서 성능 측정 (둘 다 사람 단위)
  3) DeLong 검정: 사람 단위 분할과 무작위 분할의 AUC 차이 검정 (누수 대조)

사용:
  python methodology_validation.py "<경로>/Behavioral-features - per minute.xlsx"

데이터는 SWELL-KW 배포처에서 받아야 하며 이 저장소에는 없음.
난수는 전부 random_state=42 로 고정함.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ['SnKeyStrokes', 'SnChars', 'SnSpecialKeys', 'SnDirectionKeys', 'SnErrorKeys',
            'SnShortcutKeys', 'SnSpaces', 'CharactersRatio', 'ErrorKeyRatio', 'SnMouseAct',
            'SnLeftClicked', 'SnRightClicked', 'SnDoubleClicked', 'SnWheel', 'SnDragged',
            'SnMouseDistance']


def load(path):
    df = pd.read_excel(path, sheet_name="SWELLdata")
    df['Condition'] = df['Condition'].astype(str).str.strip()
    df['y'] = df['Condition'].isin(['T', 'I']).astype(int)      # 시간압박·방해 = 1
    df['PPn'] = df['PP'].str.extract(r'(\d+)').astype(int)      # 참가자 번호
    active = df[['SnKeyStrokes', 'SnMouseAct']].fillna(0).sum(axis=1)
    df = df[active >= 1].reset_index(drop=True)                 # 활동 0인 분 제외
    return df[FEATURES].values, df['y'].values, df['PPn'].values


def make_pipe():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED))


def calibration_check(X, y, g):
    """학습 60% / 보정 20% / 평가 20% 를 사람 단위로 분리."""
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED).split(X, y, g))
    Xr, yr, gr = X[rest], y[rest], g[rest]
    cal, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(Xr, yr, gr))
    model = make_pipe().fit(X[tr], y[tr])
    iso = IsotonicRegression(out_of_bounds='clip').fit(model.predict_proba(Xr[cal])[:, 1], yr[cal])
    p_te = model.predict_proba(Xr[te])[:, 1]
    return brier_score_loss(yr[te], p_te), brier_score_loss(yr[te], iso.predict(p_te))


def nested_cv(X, y, g):
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    param = {'logisticregression__C': [0.01, 0.1, 1, 10]}
    aucs = []
    for tr, te in outer.split(X, y, g):
        gs = GridSearchCV(make_pipe(), param, cv=inner, scoring='roc_auc')
        gs.fit(X[tr], y[tr], groups=g[tr])
        aucs.append(roc_auc_score(y[te], gs.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def delong_test(y_true, p1, p2):
    """두 AUC 차이에 대한 DeLong 검정. (auc1, auc2, z, p) 반환."""
    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1; i = j
        T2 = np.empty(N); T2[J] = T
        return T2
    order = (-y_true).argsort(); label = y_true[order]
    m = int(label.sum()); n = len(label) - m
    preds = np.vstack((p1, p2))[:, order]
    tx = np.array([midrank(preds[k, :m]) for k in range(2)])
    ty = np.array([midrank(preds[k, m:]) for k in range(2)])
    tz = np.array([midrank(preds[k, :]) for k in range(2)])
    auc = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    z = (auc[0] - auc[1]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc[0], auc[1], z, p


def leakage_contrast(X, y, g):
    p_group = cross_val_predict(make_pipe(), X, y, groups=g,
                                cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                                method='predict_proba')[:, 1]
    p_random = cross_val_predict(make_pipe(), X, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                 method='predict_proba')[:, 1]
    return delong_test(y, p_group, p_random)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    X, y, g = load(sys.argv[1])
    print(f"데이터 {len(y)}분, 참가자 {len(np.unique(g))}명")
    b0, b1 = calibration_check(X, y, g)
    print(f"[보정] Brier 보정 전 {b0:.4f} / 보정 후 {b1:.4f}")
    m, s = nested_cv(X, y, g)
    print(f"[Nested CV] AUC {m:.3f} +- {s:.3f}")
    a1, a2, z, p = leakage_contrast(X, y, g)
    print(f"[DeLong] 사람 단위 AUC {a1:.3f} / 무작위 분할 AUC {a2:.3f} / z {z:.2f} / p {p:.4g}")
