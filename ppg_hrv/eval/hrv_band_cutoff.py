# -*- coding: utf-8 -*-
import numpy as np, pandas as pd

PRED_PATH   = "maus_3class_loso_pred.csv"   # score = v5(서빙, 60s) LOSO OOF ×100 (maus_add_score.py)
SCORE_COL   = "score"
LABEL_COL   = "y_true"
SENS_TARGET = 0.85
SPEC_TARGET = 0.85

df = pd.read_csv(PRED_PATH, dtype={"subject": str}, encoding="utf-8-sig")

# 라벨을 이진(저=0 / 고=1)로. 3class 파일이면 저=0, 중·과=고(1)
BINARY_MAP = {0: 0, 1: 1, 2: 1}          # y_true가 0/1/2 (저/중/과)일 때
y = df[LABEL_COL].map(BINARY_MAP) if df[LABEL_COL].isin(BINARY_MAP).all() else (df[LABEL_COL] > 0).astype(int)
s = df[SCORE_COL].astype(float).values
y = y.values

lo, hi = s[y == 0], s[y == 1]            # 저부하 점수들 / 고부하 점수들

# 아래선 = 고부하의 (1-SENS_TARGET) 분위 → 고부하의 SENS_TARGET이 이 위에 있음
#   (그 아래로는 고부하가 거의 안 내려옴 → '확실한 저')
low_cut = np.quantile(hi, 1 - SENS_TARGET)

# 위선 = 고부하가 아닌(저부하) 걸 SPEC_TARGET로 걸러내는 지점
#   (저부하의 상위 (1-SPEC) 만 위선 위로 새어나감 → 그 위는 '확실한 고')
high_cut = np.quantile(lo, SPEC_TARGET)
# 저·고 분리가 약하면 low_cut>high_cut 역전 가능 → 교정
if low_cut > high_cut:
    low_cut, high_cut = sorted([low_cut, high_cut])

print(f"아래선(확실한 저 상한): {low_cut:.1f}")
print(f"위선(확실한 고 하한):   {high_cut:.1f}")
print(f"관찰구간: {low_cut:.1f} ~ {high_cut:.1f}  (폭 {high_cut-low_cut:.1f})")

# 각 구간에 실제 저/고가 얼마나 섞였나 (관찰구간이 진짜 '애매'한지 확인)
def band(a, b):
    m = (s >= a) & (s < b)
    n = m.sum()
    return n, (y[m] == 1).mean() if n else float('nan')
for name, a, b in [("확실한 저", -1, low_cut), ("관찰", low_cut, high_cut), ("확실한 고", high_cut, 101)]:
    n, hr = band(a, b)
    print(f"  {name:7s}: n={n:4d}  고부하비율={hr:.2f}")

# 검증: LOSO fold별로 두 선이 흔들리는지 (SUBJECT_COL 있으면)
if "subject" in df.columns:
    cuts = []
    for sid, gdf in df.groupby("subject"):
        ys = (gdf[LABEL_COL].map(BINARY_MAP) if gdf[LABEL_COL].isin(BINARY_MAP).all()
              else (gdf[LABEL_COL] > 0).astype(int)).values
        ss = gdf[SCORE_COL].astype(float).values
        if (ys == 0).sum() == 0 or (ys == 1).sum() == 0: continue
        cuts.append([np.quantile(ss[ys == 1], 1 - SENS_TARGET), np.quantile(ss[ys == 0], SPEC_TARGET)])
    cuts = np.array(cuts)
    print(f"\n[LOSO 안정성] 아래선 {cuts[:,0].mean():.1f}±{cuts[:,0].std():.1f}"
          f" / 위선 {cuts[:,1].mean():.1f}±{cuts[:,1].std():.1f}")

df_out = df.copy()
df_out["band"] = np.select([s < low_cut, s < high_cut], ["저", "관찰"], "고")
df_out.to_csv("hrv_band_assigned.csv", index=False, encoding="utf-8-sig")
