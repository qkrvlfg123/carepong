# -*- coding: utf-8 -*-
"""
check_reproducible.py - 다시 실행해도 같은 값이 나오는지 확인

확인 항목
  1) 키보드·마우스 추론: validation_samples.json 의 기대값과 일치
  2) predict.py 에 내장한 계수와 model_v1.json 이 같은 값
  3) 합성 입력 전체 실행 결과가 tests/expected_demo_output.json 과 일치
  4) 같은 프로세스에서 두 번 돌린 결과가 서로 같음

사용:  python tests/check_reproducible.py            확인
       python tests/check_reproducible.py --update   기대값 파일을 현재 결과로 다시 씀

숫자는 허용 오차 1e-6 안에서 비교함. 데이터가 필요한 학습·평가 스크립트는 여기서 돌리지 않음.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))
import run_demo                                   # noqa: E402
import predict as kbd                             # noqa: E402

EXPECTED = os.path.join(ROOT, "tests", "expected_demo_output.json")
TOL = 1e-6


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= TOL
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def check_validation_samples():
    ok = True
    for s in json.load(open(run_demo.SAMPLES, encoding="utf-8")):
        r, e = kbd.predict(s["input"]), s["expected"]
        got = {"score": r["stress_score"], "prob": r["prob"], "raw_prob": r["raw_prob"],
               "level": r["level"], "imputed": r["imputed"]}
        want = {k: e[k] for k in got}
        ok &= same(got, want)
    return ok


def check_embedded_model():
    m = json.load(open(os.path.join(ROOT, "keyboard_model", "model_v1.json"), encoding="utf-8"))
    return same(m, kbd._MODEL)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    first, second = run_demo.run_all(), run_demo.run_all()
    if "--update" in sys.argv:
        with open(EXPECTED, "w", encoding="utf-8", newline="\n") as f:
            json.dump(first, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("기대값 파일을 다시 썼음:", os.path.relpath(EXPECTED, ROOT))
        return 0
    expected = json.load(open(EXPECTED, encoding="utf-8"))
    results = [
        ("키보드 예시 5건이 기대값과 일치", check_validation_samples()),
        ("predict.py 내장 계수와 model_v1.json 일치", check_embedded_model()),
        ("합성 입력 실행 결과가 기대값 파일과 일치", same(first, expected)),
        ("두 번 실행한 결과가 서로 일치", same(first, second)),
    ]
    for name, ok in results:
        print(("OK   " if ok else "FAIL ") + name)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
