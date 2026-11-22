# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FP)
plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

BLUE="#2563eb"; RED="#dc2626"; GREEN="#16a34a"; SLATE="#64748b"
os.makedirs("figs", exist_ok=True)

FEATS = ['SnKeyStrokes','SnChars','SnSpecialKeys','SnDirectionKeys','SnErrorKeys',
         'SnShortcutKeys','SnSpaces','CharactersRatio','SnMouseAct','SnLeftClicked',
         'SnRightClicked','SnDoubleClicked','SnWheel','SnDragged','SnMouseDistance','ErrorKeyRatio']
KO = {'SnKeyStrokes':'키입력수','SnChars':'글자수','SnSpecialKeys':'특수키','SnDirectionKeys':'방향키',
      'SnErrorKeys':'오류키','SnShortcutKeys':'단축키','SnSpaces':'스페이스','CharactersRatio':'글자비율',
      'SnMouseAct':'마우스활동','SnLeftClicked':'좌클릭','SnRightClicked':'우클릭','SnDoubleClicked':'더블클릭',
      'SnWheel':'휠','SnDragged':'드래그','SnMouseDistance':'이동거리','ErrorKeyRatio':'오류비율'}

w = pd.read_csv("nb_analysis/windows.csv")
w = w[~w.session_id.astype(str).str.startswith("ses_smoke")].copy()
feat = w.features.apply(json.loads).apply(pd.Series)
w = pd.concat([w.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
sr = pd.read_csv("nb_analysis/self_reports.csv")
real = sr[(~sr.window_id.astype(str).str.startswith("win_smoke")) &
          (sr.status=="answered") & (sr.trigger.isin(["random","spike"]))].copy()
lab = real.merge(w, on="window_id", how="inner", suffixes=("","_w"))
y = lab.yesno.astype(int).values
X = lab[FEATS].astype(float).fillna(lab[FEATS].astype(float).median())
print("라벨:", len(lab), "yes율 %.2f" % y.mean())
konames = [KO[f] for f in FEATS]

# ================= FIG A: 상관 히트맵 (피처-피처 스피어만) =================
C = X.rank().corr(method="pearson").values   # Spearman via rank
fig, ax = plt.subplots(figsize=(8.2, 7))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(16)); ax.set_xticklabels(konames, rotation=60, ha="right", fontsize=8.5)
ax.set_yticks(range(16)); ax.set_yticklabels(konames, fontsize=8.5)
for i in range(16):
    for j in range(16):
        v = C[i, j]
        if abs(v) >= 0.5:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.7 else "#333")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("스피어만 상관", fontsize=9)
ax.set_title("피처-피처 상관 히트맵 — 입력량 계열끼리 강한 다중공선성(빨강 블록)\n(상관 상위만 쓰면 중복 피처 과점 → 16개 전부 결합이 타당)", fontsize=10.5, pad=10)
plt.savefig("figs/fig_da_heatmap.png", dpi=170); plt.close()

# ================= FIG B: 상호정보량 (비선형 연관) vs 선형 상관 =================
mi = mutual_info_classif(X.values, y, discrete_features=False, random_state=0, n_neighbors=5)
pb = np.array([abs(stats.pointbiserialr(y, X[f].values)[0]) for f in FEATS])
order = np.argsort(mi)
fig, ax = plt.subplots(figsize=(8.2, 6))
yp = np.arange(16)
ax.barh(yp-0.2, mi[order], 0.4, color=BLUE, label="상호정보량 MI (비선형 연관)")
ax.barh(yp+0.2, pb[order], 0.4, color=SLATE, label="|점이연 r| (선형 상관)")
ax.set_yticks(yp); ax.set_yticklabels([konames[i] for i in order], fontsize=9)
ax.set_xlabel("라벨과의 연관 강도")
ax.axvline(0.05, color=RED, ls="--", lw=.9, alpha=.7)
ax.set_title("상호정보량(비선형) vs 선형 상관 — 비선형 연관도 전부 약함(MI 최대 0.13)\n→ 숨은 비선형 신호 없음: 개별 피처 약함을 두 지표가 함께 확인", fontsize=10.5, pad=10)
ax.legend(loc="lower right", fontsize=9)
plt.savefig("figs/fig_da_mi.png", dpi=170); plt.close()

# ================= FIG C: PCA 2D 분리도 =================
Xz = StandardScaler().fit_transform(X.values)
p = PCA(n_components=2).fit(Xz); Z = p.transform(Xz)
ev = p.explained_variance_ratio_ * 100
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
ax.scatter(Z[y==0,0], Z[y==0,1], s=14, c=GREEN, alpha=.45, label="여유(0)", edgecolors="none")
ax.scatter(Z[y==1,0], Z[y==1,1], s=14, c=RED, alpha=.45, label="스트레스(1)", edgecolors="none")
for c, col in [(0, GREEN), (1, RED)]:
    ax.scatter(Z[y==c,0].mean(), Z[y==c,1].mean(), s=260, marker="X", c=col, edgecolors="black", linewidths=1.5, zorder=5)
ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
ax.set_xlim(np.percentile(Z[:,0],1), np.percentile(Z[:,0],99))
ax.set_ylim(np.percentile(Z[:,1],1), np.percentile(Z[:,1],99))
ax.set_title("PCA 2D — 두 집단 중심(X)은 다르나 분포가 크게 겹침", fontsize=10.5)
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=.25)
ax = axes[1]
cum = np.cumsum(PCA().fit(Xz).explained_variance_ratio_) * 100
ax.plot(range(1, 17), cum, "o-", color=BLUE, lw=1.8, ms=5)
ax.axhline(90, color=SLATE, ls="--", lw=.9); ax.text(1, 91, "90%", fontsize=8, color=SLATE)
ax.set_xlabel("주성분 개수"); ax.set_ylabel("누적 설명 분산(%)"); ax.set_ylim(0, 105)
ax.set_title("누적 설명분산 — 분산이 여러 축에 분산\n(단일 축 지배 없음 → 결합이 필요)", fontsize=10.5)
ax.grid(alpha=.25)
fig.suptitle("PCA 구조 분석 — 개별·저차원으로는 선형 분리 어려움 → 16피처 결합 모델의 근거", fontsize=11, y=1.02)
plt.savefig("figs/fig_da_pca.png", dpi=170); plt.close()

print("MI max=%.3f, PCA PC1+PC2=%.0f%%" % (mi.max(), ev[:2].sum()))
print("saved: fig_da_heatmap, fig_da_mi, fig_da_pca")
