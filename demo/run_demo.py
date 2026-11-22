# -*- coding: utf-8 -*-
"""
run_demo.py - 합성 입력으로 두 모델을 끝까지 돌려 봄

실측 데이터는 쓰지 않음.
  - 키보드·마우스: keyboard_model/validation_samples.json 의 예시 입력 5건
  - PPG: 심박과 RMSSD를 정해 놓고 만든 합성 파형 3건 (seed 고정)

사용:  python demo/run_demo.py           결과를 화면에 출력
       python demo/run_demo.py --json    결과를 JSON 으로 출력 (tests 에서 사용)
"""
import json
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "keyboard_model"))
sys.path.insert(0, os.path.join(ROOT, "ppg_hrv"))

from predict import predict                      # noqa: E402
import ppg_pipeline as pp                        # noqa: E402
from stress_judge import judge_stress            # noqa: E402
from preprocess import aggregate_30min           # noqa: E402
from personalize import personalize, shrink_weight  # noqa: E402

BUNDLE = os.path.join(ROOT, "ppg_hrv", "ppg_stress_base_maus_60s.pkl")
SAMPLES = os.path.join(ROOT, "keyboard_model", "validation_samples.json")

# (평균 심박, RMSSD ms). 60초, 30fps, seed 0
SYNTHETIC_PPG = [(72, 42), (88, 22), (65, 60)]


def run_keyboard():
    out = []
    for s in json.load(open(SAMPLES, encoding="utf-8")):
        r = predict(s["input"])
        out.append({"case": s["case"], "stress_score": r["stress_score"], "prob": r["prob"],
                    "raw_prob": r["raw_prob"], "level": r["level"], "imputed": r["imputed"]})
    return out


def run_ppg():
    out = []
    for hr, rmssd in SYNTHETIC_PPG:
        raw, fps, _truth = pp._make_synthetic_ppg(mean_hr=hr, rmssd_ms=rmssd, seconds=60.0, seed=0)
        res = pp.process_ppg(raw, fps)
        row = {"input": {"mean_hr": hr, "rmssd_ms": rmssd}, "status": res["status"]}
        if res["status"] == "ok":
            h = res["hrv"]
            row["hrv"] = {k: round(float(h[k]), 6) for k in ("mean_hr", "rmssd", "sdnn", "pnn50")}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                j = judge_stress(h, BUNDLE, personal_calm_hrv=None)
            row["judge"] = {"score": int(j["score"]), "verdict": j["verdict"],
                            "used_reference": j["used_reference"], "model_version": j["model_version"]}
        out.append(row)
    return out


def run_personalize():
    """개인화 계산 예시. 값은 전부 임의로 정한 숫자임."""
    from datetime import datetime
    from preprocess import FEATS
    base = {f: 10.0 for f in FEATS}
    rows = [("demo-user", datetime(2026, 1, 1, 9, m, 5), dict(base, SnKeyStrokes=80.0 + m)) for m in range(5)]
    rows.append(("demo-user", datetime(2026, 1, 1, 9, 0, 40), dict(base, SnKeyStrokes=1.0)))  # 같은 분 중복
    x = aggregate_30min(rows)
    g_mean = {f: 10.0 for f in FEATS}; g_mean["SnKeyStrokes"] = 60.0
    g_std = {f: 5.0 for f in FEATS}
    p_mean = {"SnKeyStrokes": 90.0}
    return {
        "aggregated_SnKeyStrokes": round(x["SnKeyStrokes"], 6),
        "w_n0": shrink_weight(0), "w_n300": shrink_weight(300), "w_n900": shrink_weight(900),
        "z_SnKeyStrokes_n0": round(personalize(x, g_mean, g_std, p_mean, n=0)[0], 6),
        "z_SnKeyStrokes_n300": round(personalize(x, g_mean, g_std, p_mean, n=300)[0], 6),
    }


def run_all():
    return {"keyboard": run_keyboard(), "ppg": run_ppg(), "personalize": run_personalize()}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = run_all()
    if "--json" in sys.argv:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("[키보드·마우스] 예시 입력 -> 스트레스 지수")
        for r in result["keyboard"]:
            print(f"  {r['case']:<28} 지수 {r['stress_score']}  단계 {r['level']}")
        print("[PPG] 합성 파형 -> HRV -> 판정")
        for r in result["ppg"]:
            if r["status"] == "ok":
                h, j = r["hrv"], r["judge"]
                print(f"  입력 HR {r['input']['mean_hr']} / RMSSD {r['input']['rmssd_ms']}"
                      f"  ->  HR {h['mean_hr']:.1f}, RMSSD {h['rmssd']:.1f}, 점수 {j['score']} ({j['verdict']})")
            else:
                print(f"  입력 HR {r['input']['mean_hr']}  ->  {r['status']}")
        print("[개인화] 임의 값으로 계산 예시")
        for k, v in result["personalize"].items():
            print(f"  {k}: {v}")
