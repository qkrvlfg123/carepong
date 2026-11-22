# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FP)
plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

BLUE="#2563eb"; RED="#dc2626"; GREEN="#16a34a"; SLATE="#64748b"; AMBER="#d97706"
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
print("라벨 조인:", len(lab), "yes율 %.2f" % y.mean())

# FIG1
rows=[]
for f in FEATS:
    x=lab[f].astype(float).values
    rows.append((f, stats.pointbiserialr(y,x)[0], stats.spearmanr(x,y)[0]))
corr=pd.DataFrame(rows,columns=["f","pb","sp"]).sort_values("pb",key=lambda s:s.abs())
fig,ax=plt.subplots(figsize=(8,6)); yp=np.arange(len(corr))
ax.barh(yp-0.2,corr.pb,0.4,color=BLUE,label="점이연 r")
ax.barh(yp+0.2,corr.sp,0.4,color=SLATE,label="스피어만 ρ")
ax.set_yticks(yp); ax.set_yticklabels([KO[f] for f in corr.f]); ax.axvline(0,color="#999",lw=.8)
for t in (0.3,-0.3): ax.axvline(t,color=RED,ls="--",lw=.9,alpha=.7)
ax.set_xlim(-0.35,0.35); ax.set_xlabel("라벨과의 상관계수")
ax.set_title("16피처-라벨 상관 (자체데이터 252건) - 전부 |r|<=0.15, 빨간선=0.3 기준")
ax.legend(loc="lower right"); plt.savefig("figs/fig1_corr.png"); plt.close()

# FIG2
oc=['SnDoubleClicked','SnRightClicked','SnDirectionKeys','SnDragged']
fig,axes=plt.subplots(1,4,figsize=(11,3.6))
for ax,f in zip(axes,oc):
    d=w[f].astype(float).values
    ax.boxplot(d,widths=0.5,flierprops=dict(marker='o',ms=2,mfc=RED,mec='none',alpha=.3))
    ax.set_title(f"{KO[f]}\n(0인 창 {(d==0).mean()*100:.0f}%)",fontsize=10); ax.set_xticks([])
fig.suptitle("이상치 - 드문 행동은 대부분 0 -> IQR이 정상값을 이상치로 오표시(착시) -> 제거 안 함",fontsize=10.5)
plt.savefig("figs/fig2_outlier.png"); plt.close()

# FIG3
fig,axes=plt.subplots(1,2,figsize=(9.5,3.6))
d=w["SnMouseDistance"].astype(float).clip(upper=w["SnMouseDistance"].quantile(.99)).values
axes[0].hist(d,bins=50,color=SLATE); axes[0].set_title("정규화 전 - 이동거리(원값, 스케일 큼)")
z=(w[FEATS].astype(float)-w[FEATS].astype(float).mean())/w[FEATS].astype(float).std()
axes[1].hist(z["SnMouseDistance"].clip(-3,5),bins=50,color=BLUE)
axes[1].set_title("정규화 후 - 평균0.표준편차1")
for a in axes: a.set_ylabel("창 수")
fig.suptitle("정규화(StandardScaler) - 스케일 큰 피처가 로지스틱 계수를 과점하지 않게 통일",fontsize=10.5)
plt.savefig("figs/fig3_scaler.png"); plt.close()

# FIG4
fig,ax=plt.subplots(figsize=(6.4,4))
labs=["키보드\n단독(9)","마우스\n단독(7)","유의 11개","전체 16개"]; vals=[0.578,0.579,0.617,0.634]
b=ax.bar(labs,vals,color=[SLATE,SLATE,AMBER,BLUE])
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.002,f"{v:.3f}",ha="center",fontsize=10)
ax.axhline(0.5,color=RED,ls="--",lw=.9,label="무작위(0.5)")
ax.set_ylim(0.5,0.66); ax.set_ylabel("AUC (사람단위)")
ax.set_title("피처 선정 - 개별은 약해도 16개 전부 결합이 최고(0.634)"); ax.legend()
plt.savefig("figs/fig4_select.png"); plt.close()

# FIG5
fig,ax=plt.subplots(figsize=(6.8,4.2))
stg=["직전 1분\n개인화X","직전 30분\n집계","30분 +\n개인화"]; av=[0.48,0.55,0.62]
b=ax.bar(stg,av,color=[SLATE,AMBER,BLUE])
for r,v in zip(b,av): ax.text(r.get_x()+r.get_width()/2,v+0.004,f"{v:.2f}",ha="center",fontsize=11)
ax.axhline(0.5,color=RED,ls="--",lw=1,label="무작위(0.5)")
ax.axhline(0.63,color=GREEN,ls=":",lw=1.3,label="SWELL 통제실험실 상한(0.63)")
ax.set_ylim(0.45,0.68); ax.set_ylabel("AUC (사람단위 LOSO)")
ax.set_title("1차 실데이터 검증 - 집계구간.개인화로 0.48->0.62 (SWELL 상한 근접)")
ax.legend(loc="upper left",fontsize=9); plt.savefig("figs/fig5_auc.png"); plt.close()

# FIG6
w2=pd.read_csv("nb_analysis/windows.csv")
w2['minute']=pd.to_datetime(w2.window_start).dt.floor('min').astype(str)
cnt=w2.groupby(['user_id','minute']).size()
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
axes[0].bar(["raw\n(전체 전송)","분단위\n중복제거"],[194484,108566],color=[SLATE,BLUE])
axes[0].text(0,194484,"194,484",ha="center",va="bottom"); axes[0].text(1,108566,"108,566",ha="center",va="bottom")
axes[0].set_title("중복 44.2% 제거"); axes[0].set_ylabel("행 수"); axes[0].set_ylim(0,220000)
vc=cnt.value_counts().sort_index(); vcp=vc[vc.index<=8]
axes[1].bar(vcp.index.astype(str),vcp.values,color=BLUE)
axes[1].set_xlabel("같은 사용자.같은 '분'에 들어온 전송 수"); axes[1].set_ylabel("해당 분 개수")
axes[1].set_title(f"탐지법: (user,분)으로 묶어 count>1 = 중복 (최대 {cnt.max()}건/분)")
fig.suptitle("중복 버그 탐지 - 같은 사람.같은 분에 여러 번 전송된 걸 묶어 발견",fontsize=10.5)
plt.savefig("figs/fig6_dup.png"); plt.close()

# FIG7
umean=w.groupby("user_id")["SnKeyStrokes"].apply(lambda s:s.astype(float).mean())
ucount=w.groupby("user_id").size(); valid=ucount[ucount>=30].index
full_ratio=umean[valid].max()/max(umean[valid].min(),0.1)
order=ucount[valid].sort_values(ascending=False).head(16).index
data=[w[w.user_id==u]["SnKeyStrokes"].astype(float).values for u in order]
fig,ax=plt.subplots(figsize=(10,4.2))
bp=ax.boxplot(data,showfliers=False,patch_artist=True,medianprops=dict(color="black"))
for p in bp['boxes']: p.set_facecolor(BLUE); p.set_alpha(.55)
ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels([f"U{i+1}" for i in range(len(order))],fontsize=8)
ax.set_ylabel("1분 창 키입력수")
ax.set_title(f"사용자별 평소 타이핑량 - 개인 간 평균 약 {full_ratio:.0f}배 차 -> '그 사람 평소 대비'(개인화) 필요")
plt.savefig("figs/fig7_peruser.png"); plt.close()
print("FIG7 ratio %.0f (%.1f~%.1f)"%(full_ratio,umean[valid].min(),umean[valid].max()))

# FIG8
X=lab[FEATS].astype(float).values; g=lab.user_id.values; Xz=X.copy()
for u in np.unique(g):
    m=g==u; mu=X[m].mean(0); sd=X[m].std(0); sd[sd==0]=1; Xz[m]=(X[m]-mu)/sd
clf=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xz,y)
pi=permutation_importance(clf,Xz,y,n_repeats=30,random_state=0,scoring="roc_auc")
imp=pd.DataFrame({"f":FEATS,"imp":pi.importances_mean}).sort_values("imp")
fig,ax=plt.subplots(figsize=(8,6))
ax.barh([KO[f] for f in imp.f],imp.imp,color=[RED if v>0 else SLATE for v in imp.imp])
ax.set_xlabel("순열 중요도 (제거 시 AUC 감소량)")
ax.set_title("자체데이터 피처 중요도 - 활동량(스페이스.키입력.휠.클릭)이 판정 주도")
plt.savefig("figs/fig8_importance.png"); plt.close()

# FIG9
w["hour"]=pd.to_datetime(w.window_start).dt.hour
act=w.groupby("hour").size()
sra=sr[(~sr.window_id.astype(str).str.startswith("win_smoke"))&(sr.status=="answered")].copy()
sra["hour"]=pd.to_datetime(sra.prompt_time).dt.hour
yr=sra.groupby("hour").yesno.mean()
fig,ax1=plt.subplots(figsize=(9.5,4))
ax1.bar(act.index,act.values,color=BLUE,alpha=.55); ax1.set_xlabel("시각(시)")
ax1.set_ylabel("활동 창 수",color=BLUE); ax2=ax1.twinx()
yrp=yr[(yr.index>=9)&(yr.index<=18)]
ax2.plot(yrp.index,yrp.values,color=RED,marker="o",lw=2); ax2.set_ylabel("스트레스 yes율",color=RED); ax2.set_ylim(0,1)
ax1.set_title("시간대 패턴 - 활동 9~18시 집중.점심(12~13시) 저점 / yes율 오후 높음")
ax1.set_xticks(range(8,20)); plt.savefig("figs/fig9_temporal.png"); plt.close()

# FIG10
m=pd.read_csv("carepong-ppg-hrv/data/maus_our_60s.csv").dropna()
HF=["mean_hr","rmssd","sdnn","pnn50"]; HKO={"mean_hr":"평균심박(bpm)","rmssd":"RMSSD(ms)","sdnn":"SDNN(ms)","pnn50":"pNN50(%)"}
fig,axes=plt.subplots(1,4,figsize=(11,3.8))
for ax,f in zip(axes,HF):
    lo=m[m.label==0][f].values; hi=m[m.label==1][f].values
    bp=ax.boxplot([lo,hi],tick_labels=["저부하","고부하"],showfliers=False,patch_artist=True,widths=0.6,medianprops=dict(color="black"))
    bp['boxes'][0].set_facecolor(GREEN); bp['boxes'][0].set_alpha(.5)
    bp['boxes'][1].set_facecolor(RED); bp['boxes'][1].set_alpha(.5)
    ax.set_title(HKO[f],fontsize=10)
fig.suptitle("2차 HRV 파생변수 - 고부하에서 심박↑.HRV(변이)↓ (생리학적 정상, MAUS 757창)",fontsize=10.5)
plt.savefig("figs/fig10_hrv_box.png"); plt.close()

# FIG11
Xm=m[HF].values; ym=m.label.values; gm=m.subject.values; Xmz=Xm.copy()
for u in np.unique(gm):
    mm=gm==u; mu=Xm[mm].mean(0); sd=Xm[mm].std(0); sd[sd==0]=1; Xmz[mm]=(Xm[mm]-mu)/sd
Xs=StandardScaler().fit_transform(Xmz)
lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xs,ym); coef=lr.coef_[0]
pim=permutation_importance(lr,Xs,ym,n_repeats=50,random_state=0,scoring="roc_auc").importances_mean
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
axes[0].bar([HKO[f] for f in HF],coef,color=[RED if c>0 else GREEN for c in coef])
axes[0].axhline(0,color="#999",lw=.8); axes[0].set_title("로지스틱 계수 (부호=방향)")
axes[1].bar([HKO[f] for f in HF],pim,color=BLUE); axes[1].set_title("순열 중요도 (기여 크기)")
fig.suptitle("2차 HRV 피처 중요도 - 심박(mean_hr)이 최대 기여, 방향 생리학적 정상",fontsize=10.5)
plt.savefig("figs/fig11_hrv_imp.png"); plt.close()
print("FIG11 coef:", dict(zip(HF,coef.round(2))))
print("DONE:", sorted(f for f in os.listdir("figs") if f.endswith(".png")))
