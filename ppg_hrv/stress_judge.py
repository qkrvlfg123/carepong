# -*- coding: utf-8 -*-
"""
stress_judge.py — 손가락 PPG 스트레스 판정 함수  (shrinkage 개인화 버전)
========================================================================
※ 판정 모델은 MAUS 학습 ppg_stress_base_maus_60s.pkl(v5: 60초 · mean_hr·rmssd·sdnn·pnn50).
   이 모듈은 그 모델로 점수·판정을 냄.

[변경점 — 콜드→개인 "서서히"(shrinkage)]
  이전: 안정 측정 n<5면 콜드(100% 인구), n≥5면 개인(100% 본인)으로 **딱 끊김**.
  변경: w = n/(n+k) 로 **점진 블렌드**. n 적으면 인구에 가깝고, 쌓일수록 본인에 가까움.
        - mu = w·개인평균 + (1-w)·인구평균
        - sd = w·개인표준편차 + (1-w)·인구표준편차   (얇을 땐 인구로 당겨 안정화)
        - 키보드(kbd_personalize)와 같은 원리. 버전업 없이 데이터 쌓일수록 개인화.
  k(shrink_k)는 "반반 지점"(안정 측정 수). 번들에 없으면 기본 10 (≈ 안정측정 10개에서 50:50).
  ※ 모델 재학습 불필요 — 입력 정규화 방식만 바뀜(모델은 pop_stats 그대로 사용).

judge_stress(hrv, bundle_path, personal_calm_hrv):
  personal_calm_hrv : 그 유저 '안정(baseline)' 측정 HRV 딕셔너리 리스트. 없으면 콜드스타트.
  반환: {score, verdict, used_reference(personal|coldstart|absolute), cutoff, model_version}
"""
import numpy as np

BAND_LOW, BAND_HIGH = 38.5, 65.7   # v5, SENS/SPEC 0.85


def judge_stress(hrv: dict, bundle_path: str = "ppg_stress_base_maus_60s.pkl",
                 personal_calm_hrv: list | None = None) -> dict:
    import joblib
    b = joblib.load(bundle_path)
    feats = b["features"]
    x = np.array([hrv[f] for f in feats], dtype=float)

    if b.get("mode") == "personal":
        pop_mu = np.array([b["pop_stats"][f][0] for f in feats], dtype=float)
        pop_sd = np.array([b["pop_stats"][f][1] for f in feats], dtype=float)
        calm = personal_calm_hrv or []
        n = len(calm)
        k = b.get("shrink_k", 10)                       # 반반 지점(안정 측정 수)
        if n > 0:
            cm = np.array([[c[f] for f in feats] for c in calm], dtype=float)
            u_mu, u_sd = cm.mean(0), cm.std(0)
            w = n / (n + k)                             # 개인 비중 (n 커질수록 →1)
            mu = w * u_mu + (1.0 - w) * pop_mu          # 평균: 개인↔인구 점진 블렌드
            sd = w * u_sd + (1.0 - w) * pop_sd          # 표준편차도 점진 블렌드
            used = "personal" if w >= 0.5 else "coldstart"   # 표시용 라벨
        else:
            mu, sd, used = pop_mu, pop_sd, "coldstart"  # 데이터 0 → 인구 공통
        sd = np.where(sd == 0, 1.0, sd)
        x = (x - mu) / sd
    else:
        used = "absolute"

    p = float(b["model"].predict_proba(x.reshape(1, -1))[0, 1])
    score = round(p * 100)
    verdict = "confirm" if score >= BAND_HIGH else ("observe" if score >= BAND_LOW else "deny")
    return {"score": score,
            "verdict": verdict,
            "used_reference": used,
            "cutoff": b["cutoff_score"],
            "model_version": b["model_version"]}
