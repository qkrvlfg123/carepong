# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score

# ===== 0) 여기만 채우기 =====================================
DATA_PATH   = "maus_hrv_60s.csv"   # maus_merge_nback.py로 22명 raw 병합 (서빙 60s 모델과 동일 창)
SUBJECT_COL = "subject"
COND_COL    = "nback"              # "0","2","3","rest" (문자열)
FEATURES    = ["mean_hr", "rmssd", "sdnn", "pnn50"]
REST_POLICY = "exclude"           # "exclude" | "merge_low"
SEED        = 42
# ==========================================================

df = pd.read_csv(DATA_PATH, dtype={SUBJECT_COL: str}, encoding="utf-8-sig")

# 1) 난이도 → 저/중/과 (0/1/2) 복원
m = {0: 0, 2: 1, 3: 2}            # 0-back=저, 2-back=중, 3-back=과
is_rest = df[COND_COL].astype(str).str.lower().eq("rest")
if REST_POLICY == "merge_low":
    df.loc[is_rest, COND_COL] = 0
else:
    df = df[~is_rest]
df[COND_COL] = pd.to_numeric(df[COND_COL])   # CSV에서 문자열로 읽힘 → 정수로
df = df[df[COND_COL].isin(m)].copy()
df["y"] = df[COND_COL].map(m)

X = df[FEATURES].values
y = df["y"].values
g = df[SUBJECT_COL].values
names = ["저", "중", "과"]

# 2) LOSO 3클래스 재학습·예측 (fold마다 train 통계로 표준화)
logo = LeaveOneGroupOut()
y_true, y_pred = [], []
for tr, te in logo.split(X, y, g):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(sc.transform(X[tr]), y[tr])
    y_true.extend(y[te]); y_pred.extend(clf.predict(sc.transform(X[te])))
y_true, y_pred = np.array(y_true), np.array(y_pred)

# 3) 검증 출력
print("=== 3클래스 혼동행렬 (행=실제, 열=예측) ===")
print(pd.DataFrame(confusion_matrix(y_true, y_pred), index=names, columns=names))
print("\nmacro-F1:", round(f1_score(y_true, y_pred, average="macro"), 3))
print(classification_report(y_true, y_pred, target_names=names, digits=3))

# 3-a) ★핵심: 중(2-back) vs 과(3-back) 만 — HRV로 갈리나?
mask = np.isin(y_true, [1, 2])
print("\n=== 중 vs 과 (핵심 질문) ===")
print(pd.DataFrame(confusion_matrix(y_true[mask], y_pred[mask], labels=[1, 2]),
                   index=["중", "과"], columns=["→중", "→과"]))

# 3-b) 순서 구조: 저↔과 직접 오분류(2칸)가 적어야 순서형 유지
far = np.mean(np.abs(y_true - y_pred) == 2)
print(f"\n저↔과 직접 오분류 비율: {far:.3f} (낮을수록 좋음)")

# 3-c) 비교군: 같은 데이터 이진(저 vs 중+과) LOSO AUC — 현 v5 0.722와 대조
yb = (y > 0).astype(int); ab = []
for tr, te in logo.split(X, yb, g):
    if len(np.unique(yb[te])) < 2: continue
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(X[tr]), yb[tr])
    ab.append(roc_auc_score(yb[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
print(f"\n이진(저 vs 중+과) LOSO AUC: {np.mean(ab):.3f} ± {np.std(ab):.3f}")

# 결과 저장
pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv("maus_3class_loso_pred.csv", index=False)
