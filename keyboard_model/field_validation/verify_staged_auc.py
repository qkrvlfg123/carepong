# -*- coding: utf-8 -*-
"""
verify_staged_auc.py - 서비스 수집 데이터에서 단계별 AUC 확인

확인하는 단계
  1) 30분 집계: 저장된 서빙 확률과 자가보고 라벨의 AUC
  2) 30분 집계 + shrinkage 개인화: 다시 학습한 LOSO AUC
  1분 창 단계는 평가하지 않음. spike 트리거가 활동이 많을 때 발동해서
  팝업이 뜬 1분 창의 피처가 라벨을 일부 설명하게 됨(트리거별 yes 비율을 출력해 확인).

전처리 규칙은 keyboard_model/preprocess.py 와 같음
  - 점검용 세션(ses_smoke) 제외
  - 같은 사용자, 같은 분 중복은 활동량이 가장 큰 1건만 남김
  - 자가보고는 random, spike 트리거 중 응답 완료, 15분 이내 응답만 사용

입력: nb_analysis/windows.csv, nb_analysis/self_reports.csv
      서비스 수집 데이터라 저장소에 없음. 외부에서는 실행할 수 없음.
사용: python verify_staged_auc.py
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

FEATS = ['SnKeyStrokes','SnChars','SnSpecialKeys','SnDirectionKeys','SnErrorKeys',
         'SnShortcutKeys','SnSpaces','CharactersRatio','SnMouseAct','SnLeftClicked',
         'SnRightClicked','SnDoubleClicked','SnWheel','SnDragged','SnMouseDistance','ErrorKeyRatio']
K = 300  # shrinkage 반반 지점 (kbd_baseline와 동일)
DATA = "nb_analysis"


def load():
    w = pd.read_csv(f"{DATA}/windows.csv")
    w = w[~w.session_id.astype(str).str.startswith("ses_smoke")].copy()
    fe = w.features.apply(lambda s: json.loads(s) if isinstance(s, str) else s).apply(pd.Series)
    for f in FEATS:
        w[f] = pd.to_numeric(fe.get(f), errors="coerce")
    w["window_start"] = pd.to_datetime(w.window_start, errors="coerce", utc=True)
    w["act"] = w[["SnKeyStrokes", "SnMouseAct"]].fillna(0).sum(axis=1)
    # ① 분단위 중복제거(같은 user·같은 분 → 활동 최대 1건)  — dedup_aggregate.py 규칙
    w["minute"] = w.window_start.dt.strftime("%Y-%m-%d %H:%M")
    w = w.sort_values("act", ascending=False).drop_duplicates(["user_id", "minute"], keep="first")
    act = w[(w.activity_zero == False) | (w.act > 0)].copy()

    sr = pd.read_csv(f"{DATA}/self_reports.csv")
    sr = sr[sr.trigger.isin(["random", "spike"]) & (sr.status == "answered")].dropna(subset=["yesno"])
    if sr.latency_sec.notna().any():
        sr = sr[sr.latency_sec.fillna(9999) <= 900]          # 15분 유효창
    sr["prompt_time"] = pd.to_datetime(sr.prompt_time, errors="coerce", utc=True)
    sr["y"] = sr.yesno.astype(float).round().astype(int)
    sr = sr.dropna(subset=["prompt_time"])
    return w, act, sr


def loso(X, y, g):
    X = np.nan_to_num(np.asarray(X, float)); oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    w, act, sr = load()
    print(f"스냅샷 활동창 {len(act):,} · 라벨 {len(sr)} (yes {sr.y.mean():.2f}) · 사용자 {sr.user_id.nunique()}")

    # ── [근거] 트리거-라벨 커플링: spike가 고활동 시 발동 → yes율이 높다 ──
    tr_yes = sr.groupby("trigger").y.mean().round(3).to_dict()
    print(f"\n[1분 제외 근거] 트리거별 yes율: {tr_yes}  → spike 커플링 존재")

    # ── (30분 집계) 저장된 서빙 prob vs 라벨 ──
    m = sr.merge(w[["window_id", "prob"]], on="window_id", how="inner").dropna(subset=["prob"])
    auc_serving = roc_auc_score(m.y, m.prob)

    # ── (30분 + 개인화) 재학습 LOSO ──
    gmean, gstd = act[FEATS].mean(), act[FEATS].std().replace(0, 1)
    umean, ucnt = act.groupby("user_id")[FEATS].mean(), act.groupby("user_id").size()
    Xc, ys, grp = [], [], []
    for _, r in sr.iterrows():
        u, t = r.user_id, r.prompt_time
        sub = act[(act.user_id == u) & (act.window_start <= t) &
                  (act.window_start >= t - pd.Timedelta(minutes=30))][FEATS]
        if len(sub) == 0:
            continue
        n = int(ucnt.get(u, 0)); wsh = n / (n + K)
        mu_eff = wsh * (umean.loc[u] if u in umean.index else gmean) + (1 - wsh) * gmean
        Xc.append(((sub.mean() - mu_eff) / gstd).values); ys.append(int(r.y)); grp.append(u)
    auc_pers = loso(Xc, np.array(ys), np.array(grp))

    print("\n=== 단계별 재현 결과 ===")
    print(f"  30분 집계  (저장된 서빙 prob) : {auc_serving:.3f}   ")
    print(f"  30분+개인화 (재학습 LOSO)     : {auc_pers:.3f}   ")
    print("  1분 raw                      : 평가 제외 (트리거 커플링, 위 근거 참조)")


if __name__ == "__main__":
    main()
