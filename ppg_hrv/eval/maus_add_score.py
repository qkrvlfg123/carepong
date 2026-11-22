# -*- coding: utf-8 -*-
"""
maus_add_score.py — maus_3class_loso_pred.csv 에 이진 모델 점수(0~100) 붙이기
=====================================================================
점수 = 서빙 이진 모델(v5: 60s · mean_hr,rmssd,sdnn,pnn50 · 개인 z-score · 로지스틱)과
       같은 레시피의 LOSO out-of-fold predict_proba × 100.
       (배포 .pkl을 그대로 쓰면 학습에 쓴 피험자를 다시 채점하는 셈 → 컷이 낙관적으로 나옴)
학습은 v5와 동일하게 rest 포함(저부하), 점수는 3class 파일의 행(rest 제외)에만 붙인다.
출력:  maus_3class_loso_pred.csv 에 subject, nback, score 컬럼 추가
"""
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from maus_window_eval import personal_z, mk

FEATS = ["mean_hr", "rmssd", "sdnn", "pnn50"]

df = pd.read_csv("maus_hrv_60s.csv", dtype={"subject": str, "nback": str}, encoding="utf-8-sig")
X = df[FEATS].values.astype(float)
y = (df["nback"].isin(["2", "3"])).astype(int).values
g = df["subject"].values
Xz = personal_z(X, g); oof = np.zeros(len(y))
for tr, te in LeaveOneGroupOut().split(Xz, y, g):
    oof[te] = mk().fit(Xz[tr], y[tr]).predict_proba(Xz[te])[:, 1]
print(f"이진 LOSO AUC (v5 재현 확인, 0.722 기대): {roc_auc_score(y, oof):.4f}")
df["score"] = np.round(oof * 100, 2)

# 3class 파일 행 순서 = rest 제외 후 피험자 정렬 순 (LeaveOneGroupOut 순서와 동일)
task = df[df["nback"] != "rest"].reset_index(drop=True)
pred = pd.read_csv("maus_3class_loso_pred.csv")
assert len(pred) == len(task)
assert (pred["y_true"].values == task["nback"].map({"0": 0, "2": 1, "3": 2}).values).all()
pred = pd.concat([task[["subject", "nback"]], pred[["y_true", "y_pred"]], task[["score"]]], axis=1)
pred.to_csv("maus_3class_loso_pred.csv", index=False, encoding="utf-8-sig")
print(pred.head()); print(pred.shape)
