#!/usr/bin/env python3
"""Pre-registered confirmation test. See PREREGISTRATION.md. Train on retry/, test on session2/."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyze import pulses, FS

HERE = Path(__file__).parent
CL = ["chat", "code", "web"]
W = 20


def load(root):
    runs = sorted(map(json.loads, open(root / "runs.jsonl")), key=lambda r: r["start"])
    runs = [r for r in runs if r["cls"] in CL]
    pk = []
    for r in runs:
        d = [(float(x["t"]), int(x["len"])) for x in csv.DictReader(open(root / "data" / r["file"])) if x["dir"] == "down"]
        pk.append((np.array([a for a, _ in d]), np.array([b for _, b in d])))
    return runs, pk, np.array([CL.index(r["cls"]) for r in runs])


def feats(t, n, seed, rate=200e6, tau=1e-3, noise=.02):
    x = pulses(t, n, W, rate)
    a = 1 / (1 + tau * FS)
    v = lfilter([a], [1, a - 1], x)
    v = v + np.random.default_rng(seed).normal(0, noise, v.shape)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    seg = np.array_split(v, 12)
    return np.r_[[s.mean() for s in seg], [s.std() for s in seg], v.mean(), v.std(), np.percentile(v, [50, 90, 99])]


def raw(t, n):
    b = np.histogram(t, np.arange(0, 21, 2), weights=n)[0].astype(float)
    return np.log1p(np.r_[b, b.sum(), len(n), np.percentile(n, [50, 90]) if len(n) else [0, 0]])


def build(runs, pk, salt, per_s=0):
    out = []
    for r, (t, n) in zip(runs, pk):
        s = r["run"] + salt
        if per_s:
            g = np.random.default_rng(s + 7)
            k = g.poisson(per_s * W)
            t, n = np.r_[t, g.uniform(0, W, k)], np.r_[n, np.full(k, 1400)]
        out.append(feats(t, n, s))
    return np.stack(out)


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=3000))


def perm_p(Xtr, ytr, Xte, yte, acc, n=1000):
    g = np.random.default_rng(0)
    null = [(mk().fit(Xtr, g.permutation(ytr)).predict(Xte) == yte).mean() for _ in range(n)]
    return float((1 + sum(v >= acc for v in null)) / (n + 1)), float(np.mean(null))


rtr, ptr, ytr = load(HERE / "retry")
rte, pte, yte = load(HERE / "session2")
assert len(rtr) == 90, len(rtr)
result = {"train_runs": len(rtr), "test_runs": len(rte), "failed_agent_exit": sum(r["agent_rc"] not in (0, None) for r in rte)}

Xtr, Xte = build(rtr, ptr, 0), build(rte, pte, 100000)
clean = mk().fit(Xtr, ytr)
p = clean.predict(Xte)
acc = float((p == yte).mean())
pv, null = perm_p(Xtr, ytr, Xte, yte, acc)
result["primary"] = dict(acc=acc, shuffled=null, p=pv, passed=pv < .05, confusion=confusion_matrix(yte, p).tolist())
print("PRIMARY simulated", result["primary"], flush=True)

Rtr = np.stack([raw(*x) for x in ptr]); Rte = np.stack([raw(*x) for x in pte])
result["real_packets_acc"] = float((mk().fit(Rtr, ytr).predict(Rte) == yte).mean())
print("real packets acc", result["real_packets_acc"], flush=True)

result["defense"] = []
g = np.random.default_rng(1)
for ps in [50, 200, 1000]:
    Dte = build(rte, pte, 100000, ps)
    h0 = clean.predict(Dte) == yte
    Dtr = build(rtr, ptr, 0, ps)
    h1 = mk().fit(Dtr, ytr).predict(Dte) == yte
    idx = g.integers(0, len(yte), (5000, len(yte)))
    diff = (h1[idx].mean(1) - h0[idx].mean(1))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    pr, nr = perm_p(Dtr, ytr, Dte, yte, float(h1.mean()))
    row = dict(per_s=ps, not_retrained=float(h0.mean()), retrained=float(h1.mean()), gain=float(h1.mean() - h0.mean()),
               ci=[float(lo), float(hi)], retrained_p=pr, shuffled=nr, supported=bool(lo > 0 and pr < .05), bonferroni=pr < .0167)
    result["defense"].append(row)
    print("DEFENSE", row, flush=True)
(HERE / "session2" / "confirm_results.json").write_text(json.dumps(result, indent=1))
