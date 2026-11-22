# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
FP="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FP); plt.rcParams["font.family"]=fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"]=False; plt.rcParams["figure.dpi"]=130
plt.rcParams["savefig.bbox"]="tight"; plt.rcParams["axes.grid"]=True; plt.rcParams["grid.alpha"]=.25
BLUE="#2563eb"; RED="#dc2626"; GREEN="#16a34a"; SLATE="#64748b"; AMBER="#d97706"

FEATS=['SnKeyStrokes','SnChars','SnSpecialKeys','SnDirectionKeys','SnErrorKeys','SnShortcutKeys',
 'SnSpaces','CharactersRatio','SnMouseAct','SnLeftClicked','SnRightClicked','SnDoubleClicked',
 'SnWheel','SnDragged','SnMouseDistance','ErrorKeyRatio']
KO={'SnKeyStrokes':'키입력수','SnChars':'글자수','SnSpecialKeys':'특수키','SnDirectionKeys':'방향키',
 'SnErrorKeys':'오류키','SnShortcutKeys':'단축키','SnSpaces':'스페이스','CharactersRatio':'글자비율',
 'SnMouseAct':'마우스활동','SnLeftClicked':'좌클릭','SnRightClicked':'우클릭','SnDoubleClicked':'더블클릭',
 'SnWheel':'휠','SnDragged':'드래그','SnMouseDistance':'이동거리','ErrorKeyRatio':'오류비율'}
w=pd.read_csv("nb_analysis/windows.csv")
w=w[~w.session_id.astype(str).str.startswith("ses_smoke")].copy()
feat=w.features.apply(json.loads).apply(pd.Series)
w=pd.concat([w.reset_index(drop=True),feat.reset_index(drop=True)],axis=1)

# ========= FIG2 재작성: 이상치 설명 붙인 박스플롯 =========
f='SnDoubleClicked'
d=w[f].astype(float).values
fig,ax=plt.subplots(figsize=(7.5,4.2))
bp=ax.boxplot(d,vert=False,widths=0.5,patch_artist=True,
    flierprops=dict(marker='o',ms=3,mfc=RED,mec='none',alpha=.35),
    medianprops=dict(color="black",lw=2),boxprops=dict(facecolor=BLUE,alpha=.5))
ax.set_yticks([]); ax.set_xlabel("1분 창당 더블클릭 수")
ax.set_title(f"이상치 '착시' 예시 — 더블클릭 (전체 창의 {(d==0).mean()*100:.0f}%가 0)")
ax.annotate("박스=가운데 50%\n(여기선 0에 붙음)",xy=(0,1),xytext=(2.5,1.28),fontsize=9,
    arrowprops=dict(arrowstyle="->",color=SLATE))
ax.annotate("빨간 점 = 공식(IQR)이\n'이상치'로 찍은 값\n→ 실제론 정상 사용(가끔 더블클릭)",
    xy=(d.max()*0.6,1),xytext=(d.max()*0.35,0.62),fontsize=9,color=RED,
    arrowprops=dict(arrowstyle="->",color=RED))
ax.set_ylim(0.5,1.5)
fig.text(0.5,-0.02,"→ 0이 대부분인 분포라 IQR이 정상값을 이상치로 '오표시'. 진짜 오류 아니므로 제거하지 않고 유지.",
    ha="center",fontsize=9.5)
plt.savefig("figs/fig2_outlier.png"); plt.close(); print("fig2 재생성")

# ========= FIG3 재작성: 여러 피처 스케일 전/후 =========
show=['SnMouseDistance','SnKeyStrokes','SnWheel','ErrorKeyRatio']
fig,axes=plt.subplots(1,2,figsize=(10,4))
# 전: 원값 범위 (로그 스케일로 천지차이 보이기)
raw_ranges=[w[f].astype(float) for f in show]
axes[0].boxplot([r.values for r in raw_ranges],vert=False,showfliers=False,patch_artist=True,
    boxprops=dict(facecolor=SLATE,alpha=.5),medianprops=dict(color="black"))
axes[0].set_yticks(range(1,len(show)+1)); axes[0].set_yticklabels([KO[f] for f in show])
axes[0].set_xscale("symlog"); axes[0].set_xlabel("원값 (로그축)")
axes[0].set_title("정규화 전 — 피처마다 스케일 천지차이\n(이동거리 수만 vs 오류비율 0~1)")
# 후: 표준화
z=(w[show].astype(float)-w[show].astype(float).mean())/w[show].astype(float).std()
axes[1].boxplot([z[f].clip(-4,4).values for f in show],vert=False,showfliers=False,patch_artist=True,
    boxprops=dict(facecolor=BLUE,alpha=.5),medianprops=dict(color="black"))
axes[1].set_yticks(range(1,len(show)+1)); axes[1].set_yticklabels([KO[f] for f in show])
axes[1].set_xlabel("표준화 값 (평균0·표준편차1)"); axes[1].set_xlim(-4,4)
axes[1].set_title("정규화 후 — 모든 피처가 같은 범위로\n→ 스케일 큰 피처의 계수 과점 방지")
plt.savefig("figs/fig3_scaler.png"); plt.close(); print("fig3 재생성")

# ========= FIG7 재작성: 전체 사용자 =========
umean=w.groupby("user_id")["SnKeyStrokes"].apply(lambda s:s.astype(float).mean())
ucount=w.groupby("user_id").size(); valid=ucount[ucount>=30].index
order=umean[valid].sort_values().index
data=[w[w.user_id==u]["SnKeyStrokes"].astype(float).values for u in order]
ratio=umean[valid].max()/max(umean[valid].min(),0.1)
fig,ax=plt.subplots(figsize=(max(10,len(order)*0.42),4.6))
bp=ax.boxplot(data,showfliers=False,patch_artist=True,medianprops=dict(color="black"),widths=0.6)
for p in bp['boxes']: p.set_facecolor(BLUE); p.set_alpha(.55)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels([f"U{i+1}" for i in range(len(order))],fontsize=7,rotation=90)
ax.set_ylabel("1분 창 키입력수"); ax.set_xlabel(f"실사용자 {len(order)}명 (평소 타이핑량 오름차순)")
ax.set_title(f"사용자별 평소 타이핑량 — 실사용자 {len(order)}명 전체, 개인 간 평균 약 {ratio:.0f}배 차 → 개인화 필요")
plt.savefig("figs/fig7_peruser.png"); plt.close(); print(f"fig7 재생성: {len(order)}명, {ratio:.0f}배")

# ========= FIG12: 리듬 파생변수 순열검정 =========
sr=pd.read_csv("nb_analysis/self_reports.csv")
real=sr[(~sr.window_id.astype(str).str.startswith("win_smoke"))&(sr.status=="answered")&(sr.trigger.isin(["random","spike"]))].copy()
lab=real.merge(w,on="window_id",how="inner",suffixes=("","_w"))
lab["minute"]=pd.to_datetime(lab.window_start).dt.floor("min")
rv=pd.read_csv("nb_analysis/rhythm_windows_v2.csv")
rv.columns=[c.strip().replace("﻿","") for c in rv.columns]
rv["minute"]=pd.to_datetime(rv.window_start).dt.floor("min")
RF=[c for c in ['iki_mean','iki_std','iki_cv','iki_p10','iki_p90','iki_iqr','logiki_std','iki_ac1',
    'burst_ratio','fast_ratio','slow_ratio','pause_total_s','backspace_rate','imm_corr_ratio'] if c in rv.columns]
mg=lab.merge(rv[["user_id","minute"]+RF],on=["user_id","minute"],how="inner").dropna(subset=RF)
print("리듬 조인:",len(mg),"라벨, 리듬피처",len(RF))

def loso_auc(X,yv,gv):
    oof=np.zeros(len(yv))
    for tr,te in LeaveOneGroupOut().split(X,yv,gv):
        if len(np.unique(yv[tr]))<2: oof[te]=yv[tr].mean(); continue
        oof[te]=LogisticRegression(max_iter=1000,class_weight="balanced").fit(X[tr],yv[tr]).predict_proba(X[te])[:,1]
    return roc_auc_score(yv,oof)
def pz(X,gv):
    Xz=X.copy()
    for u in np.unique(gv):
        m=gv==u; mu=X[m].mean(0); sd=X[m].std(0); sd[sd==0]=1; Xz[m]=(X[m]-mu)/sd
    return Xz

if len(mg)>=60:
    yv=mg.yesno.astype(int).values; gv=mg.user_id.values
    X16=pz(mg[FEATS].astype(float).values,gv)
    X37=pz(mg[FEATS+RF].astype(float).values,gv)
    obs=loso_auc(X37,yv,gv)-loso_auc(X16,yv,gv)
    rng=np.random.default_rng(0); null=[]
    for _ in range(300):
        yp=yv.copy()
        for u in np.unique(gv):
            m=gv==u; yp[m]=rng.permutation(yp[m])
        null.append(loso_auc(X37,yp,gv)-loso_auc(X16,yp,gv))
    null=np.array(null); p=(np.sum(null>=obs)+1)/(len(null)+1)
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.hist(null,bins=30,color=SLATE,alpha=.6,label="라벨 무작위 섞었을 때 Δ(거품) 분포")
    ax.axvline(obs,color=RED,lw=2.5,label=f"실제 관측 Δ = {obs:+.3f}")
    ax.axvline(0,color="#999",lw=1,ls="--")
    ax.set_xlabel("리듬 37피처 추가 시 AUC 변화량 Δ (리듬 포함 − 미포함)")
    ax.set_ylabel("빈도")
    ax.set_title(f"리듬 파생변수 순열검정 — 관측 Δ가 '우연 분포' 안에 묻힘 (p={p:.2f}) → 효과 없음(기각)")
    ax.legend(fontsize=9)
    plt.savefig("figs/fig12_rhythm_perm.png"); plt.close()
    print(f"fig12 생성: obs Δ={obs:+.3f}, p={p:.3f}")
else:
    print("리듬 조인 부족, fig12 스킵")
print("DONE2")
