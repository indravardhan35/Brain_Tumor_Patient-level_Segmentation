import argparse, csv
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="baseline")
ap.add_argument("--B", type=int, default=5000)
args = ap.parse_args()
NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}

def load(mode):
    rows = []
    for f in range(5):
        with open(f"cv_results/{args.model}_{mode}_fold{f}.csv") as fh:
            for r in csv.DictReader(fh):
                rows.append((f, r["patient_id"], int(r["label"]), float(r["dice"])))
    return rows

pat, rnd = load("patient"), load("random")
assert len(pat) == len(rnd), "protocols must test the same slices"

def per_patient(rows):
    s, n, lab = defaultdict(float), defaultdict(int), {}
    for _, pid, l, d in rows:
        s[pid] += d; n[pid] += 1; lab[pid] = l
    return s, n, lab

sp, npat, lab = per_patient(pat)
sr, nrnd, _ = per_patient(rnd)
pids = sorted(sp)
assert all(npat[p] == nrnd[p] for p in pids)
S_pat = np.array([sp[p] for p in pids]); S_rnd = np.array([sr[p] for p in pids])
N = np.array([npat[p] for p in pids], float); L = np.array([lab[p] for p in pids])
P = len(pids)

print("Patient-grouped 5-fold, per-fold slice-mean Dice (fold0..4 | mean ± std):")
for l, nm in NAMES.items():
    v = [np.mean([d for f, _, ll, d in pat if f == k and ll == l]) for k in range(5)]
    print(f"  {nm:11s} " + " ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")

pm = (S_pat / N)[L == 2]
print(f"\nGlioma per-patient Dice (n={len(pm)}): mean {pm.mean():.3f}, median {np.median(pm):.3f}, "
      f"min {pm.min():.3f}, max {pm.max():.3f}")

groups = {**{nm: L == l for l, nm in NAMES.items()}, "overall": np.ones(P, bool)}
def sm(S, w, m):
    return (S[m] * w[m]).sum() / (N[m] * w[m]).sum()

rng = np.random.default_rng(0)
boots = {g: [] for g in groups}
for _ in range(args.B):
    w = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)
    for g, m in groups.items():
        boots[g].append((sm(S_pat, w, m), sm(S_rnd, w, m)))

def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5]); return f"[{lo:.3f}, {hi:.3f}]"

print(f"\n{'group':12s}{'n_pat':>6s}  {'patient-grouped':>26s}  {'random-slice':>26s}  {'gap (random - patient)':>26s}")
ones = np.ones(P)
for g, m in groups.items():
    b = np.array(boots[g])
    a, r = sm(S_pat, ones, m), sm(S_rnd, ones, m)
    print(f"{g:12s}{int(m.sum()):6d}  {a:.4f} {ci(b[:,0]):>18s}  {r:.4f} {ci(b[:,1]):>18s}  "
          f"{r-a:+.4f} {ci(b[:,1]-b[:,0]):>18s}")