# -*- coding: utf-8 -*-
"""
preprocess.py - 키보드·마우스 1분 창 전처리 (중복 제거, 30분 집계, 라벨 유효성)

DB 접속부 없이 계산 함수만 둠. 입력은 메모리에 올린 행 리스트.

규칙
  1) 같은 사용자, 같은 분(minute)에 여러 건이 들어오면 활동량이 가장 큰 1건만 남김.
     중복을 그대로 두면 같은 창이 학습·평가에 함께 들어가 성능이 부풀려짐.
  2) 활동이 있는 창만 모아 직전 30분 평균(16피처)을 만듦.
     과부하 알람 규칙(30분)과 집계 단위를 맞춤.
  3) 자가보고는 팝업 표시 후 15분 안에 응답한 것만 학습 라벨로 씀.
     늦은 응답은 회상 편향이 생기므로 제외함.
"""
import json
from collections import Counter

FEATS = ['SnKeyStrokes', 'SnChars', 'SnSpecialKeys', 'SnDirectionKeys', 'SnErrorKeys',
         'SnShortcutKeys', 'SnSpaces', 'CharactersRatio', 'SnMouseAct', 'SnLeftClicked',
         'SnRightClicked', 'SnDoubleClicked', 'SnWheel', 'SnDragged', 'SnMouseDistance',
         'ErrorKeyRatio']

WINDOW_MIN = 30          # 집계 구간(분)
VALID_WINDOW_MIN = 15    # 자가보고 유효 응답 지연 상한(분)
STATUS_ALL = ("answered", "pending", "dismissed", "ignored")


def expand_features(features):
    """dict 또는 JSON 문자열 -> 16피처 숫자 dict. 빠진 값은 None."""
    if isinstance(features, str):
        features = json.loads(features)
    return {f: (float(features[f]) if features.get(f) is not None else None) for f in FEATS}


def is_activity(fdict):
    """키·마우스 중 하나라도 0보다 크면 활동 창으로 봄."""
    for f in ('SnKeyStrokes', 'SnMouseAct', 'SnChars', 'SnLeftClicked'):
        v = fdict.get(f)
        if v is not None and v > 0:
            return True
    return False


def dedup_minute(rows):
    """
    분 단위 중복 제거.
    rows: [(user_id, window_start(datetime), features)]
    반환: [(user_id, 분 단위로 자른 시각, 16피처 dict)]
    """
    best = {}
    for uid, ts, feats in rows:
        key = (str(uid), ts.replace(second=0, microsecond=0))
        fd = expand_features(feats)
        score = sum(v for f in ('SnKeyStrokes', 'SnMouseAct') if (v := fd.get(f)) is not None)
        if key not in best or score > best[key][0]:
            best[key] = (score, (uid, key[1], fd))
    return [v[1] for v in best.values()]


def aggregate_30min(rows):
    """
    한 사용자의 직전 30분 행을 받아 활동 창 평균(16피처 dict)을 돌려줌.
    16피처 중 하나라도 값이 없으면 None.
    """
    acc = {f: [] for f in FEATS}
    for _uid, _ts, fd in dedup_minute(rows):
        if not is_activity(fd):
            continue
        for f in FEATS:
            if fd[f] is not None:
                acc[f].append(fd[f])
    out = {f: (sum(v) / len(v)) for f, v in acc.items() if v}
    return out if len(out) == len(FEATS) else None


def is_valid_label(shown_at, answered_at, limit_min=VALID_WINDOW_MIN):
    """팝업 표시부터 응답까지 limit_min 분 이내면 True. 미응답은 False."""
    if shown_at is None or answered_at is None:
        return False
    delay_sec = (answered_at - shown_at).total_seconds()
    return 0 <= delay_sec <= limit_min * 60


def compliance_rates(statuses):
    """status 목록 -> 상태별 건수, 응답률, 이탈률."""
    c = Counter(str(s).lower() for s in statuses)
    total = sum(c.values()) or 1
    out = {s: c.get(s, 0) for s in STATUS_ALL}
    out["total"] = total
    out["answer_rate"] = round(c.get("answered", 0) / total, 3)
    out["dropout_rate"] = round((c.get("dismissed", 0) + c.get("ignored", 0)) / total, 3)
    return out
