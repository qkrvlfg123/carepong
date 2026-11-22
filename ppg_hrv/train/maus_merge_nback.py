# -*- coding: utf-8 -*-
"""
maus_merge_nback.py — 22명 raw를 한 파일로 병합 (난이도 0/2/3/rest 보존)
=====================================================================
maus_reprocess.py(v3)와 동일 처리(반전·relaxed SQI·비중첩 창)지만,
label을 이진으로 묶지 않고 nback 원값(0/2/3, 휴식은 "rest")을 남긴다.
사용:  python maus_merge_nback.py <raw_root>
출력:  maus_hrv_30s.csv / _60s.csv / _120s.csv
       (subject,nback,trial,mean_hr,rmssd,sdnn,pnn50,sd1,sd2)
"""
import os, sys, glob, re
import numpy as np, pandas as pd
from maus_reprocess import FS, WINDOWS, ALLFEATS, windows


def main(raw_root):
    files = sorted(glob.glob(os.path.join(raw_root, "**", "inf_ppg.csv"), recursive=True))
    rows = {W: [] for W in WINDOWS}
    for fp in files:
        subj = os.path.basename(os.path.dirname(fp))
        df = pd.read_csv(fp)
        for col in df.columns:                       # 예: "Trial 2:2back"
            m = re.search(r"Trial (\d+):(\d)back", col)
            if not m: continue
            sig = -df[col].dropna().values.astype(float)
            for W in WINDOWS:
                for h in windows(sig, FS, W):
                    rows[W].append({"subject": subj, "nback": m.group(2), "trial": int(m.group(1)),
                                    **{f: round(h.get(f, np.nan), 3) for f in ALLFEATS}})
        rst = os.path.join(os.path.dirname(fp), "inf_resting.csv")
        if os.path.exists(rst):
            r = pd.read_csv(rst)
            if "Resting_PPG" in r.columns and r["Resting_PPG"].dropna().size > FS * 30:
                sig = -r["Resting_PPG"].dropna().values.astype(float)
                for W in WINDOWS:
                    for h in windows(sig, FS, W):
                        rows[W].append({"subject": subj, "nback": "rest", "trial": 0,
                                        **{f: round(h.get(f, np.nan), 3) for f in ALLFEATS}})
        print(f"  {subj} 완료")
    for W in WINDOWS:
        out = pd.DataFrame(rows[W]); fn = f"maus_hrv_{W}s.csv"
        out.to_csv(fn, index=False, encoding="utf-8-sig")
        print(f"[{W}초] {fn} — {len(out)}창 / {out['subject'].nunique()}명 / "
              f"{out['nback'].value_counts().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
