import argparse, json, os, random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["patient", "random"], required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--val_frac", type=float, default=0.125)  # of the non-test part
args = ap.parse_args()

with open("patient_level_split.json") as fh:
    src = json.load(fh)
records = src["train"] + src["val"] + src["test"]
rng = random.Random(args.seed)
k = args.k
os.makedirs("cv_folds", exist_ok=True)
print("slices per label:", dict(sorted(Counter(r["label"] for r in records).items())))
print("patients:", len({r["patient_id"] for r in records}))

def pids(rs):
    return {r["patient_id"] for r in rs}

def write_fold(f, train, val, test):
    assert len(train) + len(val) + len(test) == len(records)
    overlap = len(pids(test) & (pids(train) | pids(val))) / len(pids(test))
    if args.mode == "patient":
        assert overlap == 0 and not (pids(train) & pids(val)), "patient leakage!"
    with open(f"cv_folds/{args.mode}_fold{f}.json", "w") as fh:
        json.dump({"train": train, "val": val, "test": test}, fh)
    tc = Counter(r["label"] for r in test)
    tpc = Counter(l for _, l in {(r["patient_id"], r["label"]) for r in test})
    print(f"fold {f}: slices tr/val/te {len(train)}/{len(val)}/{len(test)} | "
          f"test slices {dict(sorted(tc.items()))} | test patients {dict(sorted(tpc.items()))} | "
          f"test patients also in train/val: {overlap:.0%}")

if args.mode == "patient":
    by_pid = defaultdict(list)
    for r in records:
        by_pid[r["patient_id"]].append(r)
    plabel = {}
    for pid, rs in by_pid.items():
        labs = {r["label"] for r in rs}
        assert len(labs) == 1, f"patient {pid} has labels {labs}"
        plabel[pid] = labs.pop()
    labels = sorted(set(plabel.values()))
    folds, c = [[] for _ in range(k)], 0
    for lab in labels:
        ps = sorted(p for p, l in plabel.items() if l == lab)
        rng.shuffle(ps)
        for p in ps:
            folds[c % k].append(p)
            c += 1
    for f in range(k):
        test_p, val_p = set(folds[f]), set()
        for lab in labels:
            rest = sorted(p for p, l in plabel.items() if l == lab and p not in test_p)
            rng.shuffle(rest)
            val_p.update(rest[:max(1, round(len(rest) * args.val_frac))])
        train = [r for r in records if r["patient_id"] not in (test_p | val_p)]
        val = [r for r in records if r["patient_id"] in val_p]
        test = [r for r in records if r["patient_id"] in test_p]
        write_fold(f, train, val, test)
else:
    idx = list(range(len(records)))
    rng.shuffle(idx)
    fold_of = {i: n % k for n, i in enumerate(idx)}
    for f in range(k):
        test = [records[i] for i in range(len(records)) if fold_of[i] == f]
        rest = [i for i in range(len(records)) if fold_of[i] != f]
        rng.shuffle(rest)
        n_val = round(len(rest) * args.val_frac)
        val = [records[i] for i in sorted(rest[:n_val])]
        train = [records[i] for i in rest[n_val:]]
        write_fold(f, train, val, test)