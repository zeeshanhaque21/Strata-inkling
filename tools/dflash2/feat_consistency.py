#!/usr/bin/env python3
"""Compare the target features the drafter received during speculative decoding (INK2_DF2_DUMPALL file A) with the features a teacher-forced prefill of the same tokens produces (file B).
usage: feat_consistency.py A.bin B.bin     (records: int32 pos0, n, then n x 4 x 2304 float32)"""
import sys
import numpy as np
H = 2304
def read(path):
    rows = {}
    raw = np.fromfile(path, np.uint8); i = 0
    while i < raw.size:
        pos0, n = np.frombuffer(raw[i:i + 8], np.int32); i += 8
        f = np.frombuffer(raw[i:i + n * 4 * H * 4], np.float32).reshape(n, 4, H); i += n * 4 * H * 4
        for r in range(n): rows[int(pos0) + r] = f[r]     # a later write of a position (re-verify) wins
    return rows
A, B = read(sys.argv[1]), read(sys.argv[2])
common = sorted(set(A) & set(B))
print(f"A rows {len(A)}, B rows {len(B)}, common {len(common)}")
names = ["visit 5", "visit 23", "visit 41", "final norm"]
err = np.array([[np.linalg.norm(A[p][t] - B[p][t]) / (np.linalg.norm(B[p][t]) + 1e-30) for t in range(4)] for p in common])
for t in range(4):
    print(f"tap {names[t]:10s} rel err: median {np.median(err[:, t]):.2e}  p90 {np.percentile(err[:, t], 90):.2e}  max {err[:, t].max():.2e}")
bad = [p for p, e in zip(common, err) if e.max() > 5e-2]
print(f"positions with any tap rel err > 5%: {len(bad)} of {len(common)}", bad[:20])
