# -*- coding: utf-8 -*-
"""
train_v1_model.py - 키보드·마우스 공통 모델(v1) 학습

SWELL-KW 분당 행동 피처 16개로 로지스틱 회귀를 학습하고 계수를 JSON으로 저장함.
  1) 라벨: 조건이 시간압박(T) 또는 방해(I)이면 1
  2) 활동이 0인 분 제외
  3) 사람 단위 5겹 교차검증으로 AUC 확인
  4) 교차검증 확률로 Isotonic 보정 후 Youden 지점을 개입 컷으로 정함
  5) 전체 데이터로 최종 모델과 보정기를 학습해 저장

사용:
  python train_v1_model.py "<경로>/Behavioral-features - per minute.xlsx" [출력=model_v1_new.json]

데이터는 SWELL-KW 배포처에서 받아야 하며 이 저장소에는 없음.
난수는 random_state=42 로 고정함.
출력 JSON의 키는 keyboard_model/model_v1.json 과 같음.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
KEYBOARD = ["SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys", "SnErrorKeys",
            "SnShortcutKeys", "SnSpaces", "CharactersRatio", "ErrorKeyRatio"]
MOUSE = ["SnMouseAct", "SnLeftClicked", "SnRightClicked", "SnDoubleClicked",
         "SnWheel", "SnDragged", "SnMouseDistance"]
FEATURES = KEYBOARD + MOUSE
CUT_REPORT = 50


def load(path):
    df = pd.read_excel(path, sheet_name="SWELLdata")
    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["y"] = df["Condition"].isin(["T", "I"]).astype(int)
    df["PPn"] = df["PP"].astype(str).str.extract(r"(\d+)").astype(int)
    active = df[["SnKeyStrokes", "SnMouseAct"]].fillna(0).sum(axis=1)
    df = df[active >= 1].reset_index(drop=True)
    return df[FEATURES].values, df["y"].values, df["PPn"].values


def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED))


def main(path, out):
    X, y, g = load(path)
    print(f"데이터 {len(y)}분, 참가자 {len(np.unique(g))}명, 라벨 1 비율 {y.mean():.3f}")

    # 사람 단위 교차검증
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    prob = cross_val_predict(make_model(), X, y, groups=g, cv=cv, method="predict_proba")[:, 1]
    print(f"사람 단위 검증 AUC {roc_auc_score(y, prob):.3f}, F1 {f1_score(y, prob >= 0.5):.3f}")

    # 지수와 개입 컷
    iso_cv = IsotonicRegression(out_of_bounds="clip").fit(prob, y)
    idx = np.round(iso_cv.predict(prob) * 100).astype(int)
    fpr, tpr, thr = roc_curve(y, idx)
    cut_intervene = int(thr[int(np.argmax(tpr - fpr))])
    print(f"지수 범위 {idx.min()}~{idx.max()}, 관찰 컷 {CUT_REPORT}, 개입 컷 {cut_intervene}")

    # 최종 학습
    final = make_model().fit(X, y)
    imputer, scaler, clf = final[0], final[1], final[2]
    final_iso = IsotonicRegression(out_of_bounds="clip").fit(final.predict_proba(X)[:, 1], y)

    model = {
        "features": FEATURES,
        "impute_median": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "iso_x": final_iso.X_thresholds_.tolist(),
        "iso_y": final_iso.y_thresholds_.tolist(),
        "cut_report": CUT_REPORT,
        "cut_intervene": cut_intervene,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(model, f)
    print("저장:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "model_v1_new.json")
