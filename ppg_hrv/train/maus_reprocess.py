# -*- coding: utf-8 -*-
"""
maus_reprocess.py — MAUS 손가락 PPG raw 재처리 (v3: 모든 HRV 피처 저장)
=====================================================================
v3: 피처 조합 실험을 위해 mean_hr·rmssd·sdnn·pnn50·sd1·sd2 를 모두 저장.
    (v2의 SQI 왜도 완화 + resting 저부하 보강은 유지)
MAUS 손가락 PPG는 반전(밸리 기반) → -1 곱함. 0back=저부하 / 2·3back=고부하 / resting=저부하.
사용:  python maus_reprocess.py <raw_root>   (같은 폴더에 ppg_pipeline.py 필요)
출력:  maus_our_30s.csv / _60s.csv / _120s.csv
       (subject,label,source,mean_hr,rmssd,sdnn,pnn50,sd1,sd2)
"""
import os, sys, glob
import numpy as np, pandas as pd
import ppg_pipeline as pp

FS = 256
WINDOWS = [30, 60, 120]
ALLFEATS = ["mean_hr", "rmssd", "sdnn", "pnn50", "sd1", "sd2"]
LABEL_MAP = {"0back": 0, "2back": 1, "3back": 1}

def get_label(col):
    for k, v in LABEL_MAP.items():
        if k in col: return v
    return None

def process_relaxed(raw, fs):
    try:
        sig, nfs = pp.preprocess(raw, fs)
        sqi = pp.compute_sqi(sig, nfs)
        hr = sqi.get("hr_spectral_est", 60)
        if not (sqi["band_power_ratio"] > 0.4 and 40 <= hr <= 180): return None
        peaks = pp.detect_peaks(sig, nfs)
        rr, corr = pp.correct_rr(pp.compute_rr(peaks, nfs))
        if rr.size < 5 or corr.get("rejected"): return None
        h = {}; h.update(pp.time_domain_hrv(rr)); h.update(pp.poincare_hrv(rr))
        return h
    except Exception:
        return None

def windows(sig, fs, W):
    n = W * fs
    for s in range(0, len(sig) - n + 1, n):
        h = process_relaxed(sig[s:s + n], fs)
        if h: yield h

def main(raw_root):
    files = sorted(glob.glob(os.path.join(raw_root, "**", "inf_ppg.csv"), recursive=True))
    if not files:
        print(f"[오류] {raw_root} 아래 inf_ppg.csv 없음"); return
    print(f"대상 {len(files)}명")
    rows = {W: [] for W in WINDOWS}
    for fp in files:
        subj = os.path.basename(os.path.dirname(fp))
        try:
            df = pd.read_csv(fp)
            for col in df.columns:
                lab = get_label(col)
                if lab is None: continue
                sig = -df[col].dropna().values.astype(float)
                for W in WINDOWS:
                    for h in windows(sig, FS, W):
                        rows[W].append({"subject": subj, "label": lab, "source": "task",
                                        **{f: round(h.get(f, np.nan), 3) for f in ALLFEATS}})
        except Exception as e:
            print(f"  {subj}: inf_ppg 실패 {e}")
        rst = os.path.join(os.path.dirname(fp), "inf_resting.csv")
        if os.path.exists(rst):
            try:
                r = pd.read_csv(rst)
                if "Resting_PPG" in r.columns and r["Resting_PPG"].dropna().size > FS * 30:
                    sig = -r["Resting_PPG"].dropna().values.astype(float)
                    for W in WINDOWS:
                        for h in windows(sig, FS, W):
                            rows[W].append({"subject": subj, "label": 0, "source": "rest",
                                            **{f: round(h.get(f, np.nan), 3) for f in ALLFEATS}})
            except Exception as e:
                print(f"  {subj}: resting 실패 {e}")
        print(f"  {subj} 완료")
    for W in WINDOWS:
        out = pd.DataFrame(rows[W]); fn = f"maus_our_{W}s.csv"
        out.to_csv(fn, index=False, encoding="utf-8-sig")
        if len(out):
            print(f"[{W}초] {fn} — {len(out)}창 / {out['subject'].nunique()}명 / "
                  f"고부하 {(out.label==1).sum()} · 저부하 {(out.label==0).sum()} (rest {(out.source=='rest').sum()})")
        else:
            print(f"[{W}초] 0창")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
