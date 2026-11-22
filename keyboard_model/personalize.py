# -*- coding: utf-8 -*-
"""
personalize.py - 키보드·마우스 피처의 shrinkage 개인화

z = (x - mu_eff) / global_std
mu_eff = w * 개인평균 + (1 - w) * 전체평균,  w = n / (n + k)

- 중심은 개인 평균 쪽으로 옮기고, 스케일은 전체 공통 표준편차를 씀.
- n(그 사용자의 평소 창 수)이 커질수록 w가 1에 가까워짐.
- 기준선이 없는 신규 사용자는 n = 0 이라 전체 평균만 씀(coldstart).

ppg_hrv/stress_judge.py 의 개인 기준선 블렌드와 같은 원리임.
이 모듈은 계산만 하며, 기준선 값은 호출하는 쪽에서 넘겨줌.
"""
from preprocess import FEATS

DEFAULT_K = 300   # 개인과 전체가 반반이 되는 창 수


def shrink_weight(n, k=DEFAULT_K):
    """개인 비중 w = n / (n + k). n 이 0 이면 0."""
    return n / (n + k) if n else 0.0


def personalize(x, global_mean, global_std, personal_mean=None, n=0, k=DEFAULT_K):
    """
    x             : 직전 30분 평균 16피처 dict (preprocess.aggregate_30min 결과)
    global_mean   : 전체 사용자 평균 dict
    global_std    : 전체 사용자 표준편차 dict
    personal_mean : 그 사용자의 평소 평균 dict. 없으면 None
    n             : 그 사용자의 평소 창 수
    반환          : FEATS 순서의 z 벡터(list[float])
    """
    personal_mean = personal_mean or {}
    w = shrink_weight(n, k)
    out = []
    for f in FEATS:
        pm = personal_mean.get(f, global_mean[f])
        mu_eff = w * pm + (1 - w) * global_mean[f]
        sd = global_std[f] if global_std[f] else 1.0
        out.append((x[f] - mu_eff) / sd)
    return out
