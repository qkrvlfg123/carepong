"""
케어퐁 키보드·마우스 컨디션 모델 — 서버 추론용 단일 파일
predict(features: dict) -> {"stress_score": 0~100 또는 None, "prob": ...}

- 외부 패키지 불필요(표준 라이브러리 json/math만). sklearn·numpy 불필요.
- 모델(계수·표준화·isotonic·컷)을 이 파일 안에 내장 → 이 파일 하나면 동작.
- 피처명은 SWELL-KW 컬럼명과 동일(판정 16개).
- 활동0(키+마우스 합 0)이면 score=None 반환.

사용:
    from predict import predict
    out = predict({"SnKeyStrokes": 85, "SnChars": 53, "SnLeftClicked": 7})
    # -> {"stress_score": 65.4, "prob": 0.655, "level": "intervene", "imputed": [...]}
"""
import math

# ── 모델 내장 (model_v1.json 동일 값) ─────────────────────────────
_MODEL = {"features": ["SnKeyStrokes", "SnChars", "SnSpecialKeys", "SnDirectionKeys", "SnErrorKeys", "SnShortcutKeys", "SnSpaces", "CharactersRatio", "ErrorKeyRatio", "SnMouseAct", "SnLeftClicked", "SnRightClicked", "SnDoubleClicked", "SnWheel", "SnDragged", "SnMouseDistance"], "impute_median": [69.0, 40.0, 27.0, 0.0, 5.0, 0.0, 8.0, 0.6117075000000001, 0.079042, 12.0, 6.0, 0.0, 0.0, 1.0, 1.0, 7161.1602995], "scaler_mean": [85.08752817430504, 53.443651389932384, 31.64387678437265, 3.3636363636363638, 9.262208865514651, 0.772351615326822, 11.138993238166792, 0.5037164552967693, 0.20990849586776858, 15.77648384673178, 7.062734785875282, 0.236664162283997, 0.43163035311795644, 3.9819684447783623, 2.0127723516153266, 10580.556444015401], "scaler_scale": [73.72970975055169, 51.17237500671023, 27.467831025474535, 11.331823237294705, 12.444331007907243, 1.877147663463453, 11.33630717844137, 0.2760002226521795, 1.3638620602079798, 14.579144969007954, 6.5271933997676985, 0.7123736203472145, 1.3557702494510526, 6.4410035595395, 2.906091282800465, 11754.269504461641], "coef": [0.41330800800601125, 0.47123795670619717, 0.23149676521598153, -0.2790637553034542, -0.12347349713071117, 0.11336488878428778, -0.46656602969994365, -0.12992916165816898, -0.04086794317034989, 0.08275945749089168, 0.708727825040094, 0.10693734332634665, -0.12386657399989003, 0.027974373650152973, -0.4382945412918213, -0.00045554089898870633], "intercept": 0.47458343189488444, "iso_x": [0.07790780337274933, 0.29583964350810854, 0.29707090172377176, 0.48285853289645, 0.4829047576458077, 0.5015088070589385, 0.5015661487847682, 0.5373681874051028, 0.5373779612543723, 0.5439783074558893, 0.5441511686136602, 0.6041364040354612, 0.6044248677370071, 0.6626771862066758, 0.6628659269263554, 0.692131306558462, 0.6921313785948714, 0.707728708461962, 0.7078529767558708, 0.7092115731656119, 0.7093908972878981, 0.9947400929034115], "iso_y": [0.3548387096774194, 0.3548387096774194, 0.42448979591836733, 0.42448979591836733, 0.52, 0.52, 0.5588235294117648, 0.5588235294117648, 0.5769230769230769, 0.5769230769230769, 0.5979643765903305, 0.5979643765903305, 0.6256281407035175, 0.6256281407035175, 0.6544502617801048, 0.6544502617801048, 0.6964285714285714, 0.6964285714285714, 0.7142857142857143, 0.7142857142857143, 0.7412280701754385, 0.7412280701754385], "cut_report": 50, "cut_intervene": 63}

FEATURES      = _MODEL["features"]          # 판정 16피처 (SWELL-KW 컬럼명)
_MEDIAN       = _MODEL["impute_median"]     # 결측 대치용 학습 중앙값
_MEAN         = _MODEL["scaler_mean"]
_SCALE        = _MODEL["scaler_scale"]
_COEF         = _MODEL["coef"]
_INTERCEPT    = _MODEL["intercept"]
_ISO_X        = _MODEL["iso_x"]
_ISO_Y        = _MODEL["iso_y"]
CUT_OBSERVE   = _MODEL["cut_report"]        # 50 (관찰 컷)
CUT_INTERVENE = _MODEL["cut_intervene"]     # 63 (개입 컷)


def _isotonic(p):
    """보정 전 확률 p -> 실제 비율에 맞춘 보정 확률(단조 보간)."""
    if p <= _ISO_X[0]:  return _ISO_Y[0]
    if p >= _ISO_X[-1]: return _ISO_Y[-1]
    for i in range(1, len(_ISO_X)):
        if p <= _ISO_X[i]:
            x0, x1 = _ISO_X[i-1], _ISO_X[i]
            y0, y1 = _ISO_Y[i-1], _ISO_Y[i]
            return y1 if x1 == x0 else y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return _ISO_Y[-1]


def predict(features: dict) -> dict:
    """
    features: {피처명: 값} dict. 일부 누락 가능(학습 중앙값으로 대치).
    반환: {"stress_score": 0~100 또는 None, "prob": 0~1 또는 None, "raw_prob",
           "level": "normal"/"observe"/"intervene" 또는 None,
           "imputed": [대치된 피처명...],
           "thresholds": {"observe":50, "intervene":63}}
    활동0(SnKeyStrokes+SnMouseAct==0)이면 score/prob/level = None.
    """
    ks = features.get("SnKeyStrokes") or 0
    ma = features.get("SnMouseAct") or 0
    if float(ks) + float(ma) == 0:
        return {"stress_score": None, "prob": None, "raw_prob": None, "level": None,
                "imputed": [], "thresholds": {"observe": CUT_OBSERVE, "intervene": CUT_INTERVENE},
                "note": "activity-zero window"}

    imputed = []
    z = _INTERCEPT
    for i, name in enumerate(FEATURES):
        v = features.get(name, None)
        if v is None:
            v = _MEDIAN[i]; imputed.append(name)
        else:
            v = float(v)
        if _SCALE[i] != 0:
            z += _COEF[i] * ((v - _MEAN[i]) / _SCALE[i])

    raw_prob = 1.0 / (1.0 + math.exp(-z))     # 로지스틱
    prob = _isotonic(raw_prob)                 # isotonic 보정
    score = round(prob * 100, 1)               # 지수 0~100

    if   score < CUT_OBSERVE:   level = "normal"
    elif score < CUT_INTERVENE: level = "observe"
    else:                        level = "intervene"

    return {"stress_score": score, "prob": round(prob, 4), "raw_prob": round(raw_prob, 4),
            "level": level, "imputed": imputed,
            "thresholds": {"observe": CUT_OBSERVE, "intervene": CUT_INTERVENE}}


if __name__ == "__main__":
    for name, f in [("평범", {k: round(_MEAN[i],1) for i,k in enumerate(FEATURES)}),
                    ("활동0", {k: 0 for k in FEATURES})]:
        print(name, predict(f)["stress_score"], predict(f)["level"])
