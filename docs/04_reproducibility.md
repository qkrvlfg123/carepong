# 재실행

## 1. 데이터 없이 확인할 수 있는 것

```bash
pip install -r requirements.txt
python tests/check_reproducible.py
```

확인 항목 4개.

1. 키보드·마우스 추론 결과가 `validation_samples.json`의 기대값과 같음
2. `predict.py`에 내장한 계수와 `model_v1.json`이 같은 값
3. 합성 입력 실행 결과가 `tests/expected_demo_output.json`과 같음
4. 같은 프로세스에서 두 번 실행한 결과가 서로 같음

- 숫자는 허용 오차 1e-6 안에서 비교함.
- 합성 PPG는 `seed=0`으로 고정함.
- 기대값 파일은 2026-11-23에 Python 3.14.5, numpy 2.5.0, scipy 1.18.0, scikit-learn 1.9.0에서 만들었고 4개 항목 모두 통과함.
- `requirements.txt`는 모델 번들을 저장한 버전(scikit-learn 1.8.0)에 맞춰 둠. 다른 버전에서는 번들을 읽을 때 버전 경고가 나옴.

## 2. 데이터를 받아야 다시 계산할 수 있는 것

### 키보드·마우스 (SWELL-KW)

```bash
python keyboard_model/train/train_v1_model.py "<경로>/Behavioral-features - per minute.xlsx" model_v1_new.json
python keyboard_model/eval/methodology_validation.py "<경로>/Behavioral-features - per minute.xlsx"
```

- `train_v1_model.py`: 사람 단위 교차검증 AUC를 출력하고 계수 JSON을 저장함. 저장소의 `model_v1.json`과 같은 키를 씀.
- `methodology_validation.py` 출력: 보정 전후 Brier, Nested CV AUC, 사람 단위와 무작위 분할의 AUC 및 DeLong 검정.
- 난수는 `random_state=42`로 고정함.
- 분석 당시 기록한 값: Brier 0.217 / 0.220, Nested CV 0.628, 사람 단위 0.621, 무작위 0.657, z = -9.50.

### PPG/HRV (MAUS)

```bash
mkdir work && cd work
export PYTHONPATH="../ppg_hrv:../ppg_hrv/train:../ppg_hrv/eval"

python ../ppg_hrv/train/maus_reprocess.py   <MAUS Raw_data 경로>   # maus_our_{30,60,120}s.csv
python ../ppg_hrv/train/maus_merge_nback.py <MAUS Raw_data 경로>   # maus_hrv_{30,60,120}s.csv
python ../ppg_hrv/train/maus_build_v5.py out.pkl ../ppg_hrv/ppg_stress_base_maus_60s.pkl
python ../ppg_hrv/eval/maus_feature_eval.py 60
python ../ppg_hrv/eval/maus_feature_eval.py 30
python ../ppg_hrv/eval/maus_window_eval.py
python ../ppg_hrv/eval/maus_3class_loso.py
python ../ppg_hrv/eval/maus_add_score.py
python ../ppg_hrv/eval/hrv_band_cutoff.py
python ../ppg_hrv/eval/paired_diff.py 60
```

- Windows PowerShell에서는 `$env:PYTHONPATH="..\ppg_hrv;..\ppg_hrv\train;..\ppg_hrv\eval"`.
- `maus_build_v5.py`는 새로 만든 번들과 저장소의 번들을 계수, 집단 통계, 컷, LOSO AUC 단위로 비교해 출력함.
- 부트스트랩은 1000회, seed 0.
- 2026-10-24에 위 순서로 실행해 출력 csv와 번들 값이 일치함을 확인한 기록이 있음. 저장소를 정리한 2026-11-23에는 데이터가 없어 다시 돌리지 않음.

## 3. 다시 계산할 수 없는 것

- `train_v1_model.py`는 학습 노트북을 스크립트로 옮긴 것임. 저장소를 정리하면서 다시 돌려 보지 못했으므로, 출력 JSON이 `model_v1.json`과 소수점까지 같은지는 확인하지 못함.
- 활동 하한선 비교, 모델 8종 비교, 개인 편차 결합(0.693), 자가보고 라벨 대조(0.482)는 보고서에 기록한 값이며 계산 코드가 이 저장소에 없음.
- 서비스 수집 데이터 분석(`keyboard_model/field_validation/`)은 코드만 있음. 데이터를 공개하지 않으므로 외부에서 다시 계산할 수 없음.
  - `verify_staged_auc.py`: 30분 집계와 개인화 단계의 AUC.
  - `02_group_comparison_rhythm.py`: 리듬 파생변수 순열검정(300회, seed 0).
  - 그림을 그리는 스크립트는 Noto Sans CJK 글꼴 경로가 리눅스 기준으로 적혀 있음.
  - 개인화 단계 AUC는 스냅샷 시점에 따라 0.585~0.62 범위로 기록되어 있음. README에는 기준 스냅샷(2026-11-02~03 재추출, 31명, 라벨 157건)의 0.619를 적음.
  - `verify_staged_auc.py`를 따로 돌렸을 때는 30분 집계 0.545, 개인화 단계 약 0.644가 나온 기록이 있음. 보고서 값 0.62의 신뢰구간(0.525~0.698) 안이며, 대표값은 보고서 기준인 0.62로 둠.

## 4. 고정한 것

| 항목 | 값 |
|---|---|
| 분할 난수 (키보드) | `random_state=42` |
| 부트스트랩 (PPG) | 1000회, seed 0 |
| 합성 PPG | seed 0, 60초, 30fps |
| 모델 번들 | `base-v5-maus-hr-60s`, scikit-learn 1.8.0 |
| 패키지 버전 | `requirements.txt` |
