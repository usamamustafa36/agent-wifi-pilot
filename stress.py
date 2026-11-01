#!/usr/bin/env python3
"""Try to break the agent-task result (simple model, simulated sniffer, retry data).

Exploratory: written after seeing the first retry scores.
  baseline      contiguous time blocks
  held-out      topics/tasks in the test fold never appear in training
  no-start      first 5 s removed (CLI start-up looks the same for every class)
  first 10 s    only the first 10 s, so a later finish time cannot help
  busy          each run overlaid with a recorded video run from a different time
  sensitivity   other assumed PHY rates, smoothing, noise
  defense       fake pulses: clean-trained vs retrained attacker
"""
import csv, json, os
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyze import pulses, FS

ROOT = Path(os.environ["PILOT_DIR"])
ALL = sorted(map(json.loads, open(ROOT / "runs.jsonl")), key=lambda r: r["start"])
CL = ["chat", "code", "web"]
W = 20


def load(r):
    d = [(float(x["t"]), int(x["len"])) for x in csv.DictReader(open(ROOT / "data" / r["file"])) if x["dir"] == "down"]
    return np.array([a for a, _ in d]), np.array([b for _, b in d])


PK = {r["run"]: load(r) for r in ALL}
RUNS = [r for r in ALL if r["cls"] in CL]
VID = [r for r in ALL if r["cls"] == "video"]
y = np.array([CL.index(r["cls"]) for r in RUNS])
groups = np.array([r["arg"] for r in RUNS])


def feats(t, n, r, rate=200e6, tau=1e-3, noise=.02, lo=0, hi=W):
    keep = (t >= lo) & (t < hi)
    x = pulses(t[keep] - lo, n[keep], hi - lo, rate)
    a = 1 / (1 + tau * FS)
    v = lfilter([a], [1, a - 1], x)
    v = v + np.random.default_rng(r["run"]).normal(0, noise, v.shape)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    seg = np.array_split(v, 12)
    return np.r_[[s.mean() for s in seg], [s.std() for s in seg], v.mean(), v.std(), np.percentile(v, [50, 90, 99])]


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=3000))


def tblock(k=5):
    for te in np.array_split(np.arange(len(y)), k):
        yield np.setdiff1d(np.arange(len(y)), te), te


def cv(X, labels=y, splits=None, Xte=None):
    pred = np.zeros(len(y), int)
    for tr, te in (splits or list(tblock())):
        pred[te] = mk().fit(X[tr], labels[tr]).predict((X if Xte is None else Xte)[te])
    return pred


def report(name, X, splits=None, Xte=None, perms=100):
    p = cv(X, splits=splits, Xte=Xte); acc = (p == y).mean()
    rng = np.random.default_rng(0)
    null = [(cv(X, rng.permutation(y), splits, Xte) == y).mean() for _ in range(perms)]
    pv = (1 + sum(v >= acc for v in null)) / (perms + 1)
    print(f"{name:38s} acc={acc:.3f} shuffled={np.mean(null):.3f} p={pv:.3f}", flush=True)
    return dict(name=name, acc=float(acc), shuffled=float(np.mean(null)), p=float(pv), confusion=confusion_matrix(y, p).tolist())


def X_of(fn=lambda r, t, n: (t, n), **kw):
    return np.stack([feats(*fn(r, *PK[r["run"]]), r, **kw) for r in RUNS])


def overlay(r, t, n):
    v = VID[r["run"] % len(VID)]
    tv, nv = PK[v["run"]]
    return np.r_[t, tv], np.r_[n, nv]


def fake(per_s):
    def f(r, t, n):
        g = np.random.default_rng(r["run"] + 7)
        k = g.poisson(per_s * W)
        return np.r_[t, np.sort(g.uniform(0, W, k))], np.r_[n, np.full(k, 1400)]
    return f


base = X_of()
out = [report("baseline", base)]
print("confusion rows=true cols=pred", CL, out[0]["confusion"], flush=True)
gk = list(GroupKFold(5).split(base, y, groups))
out.append(report("held-out topics/tasks", base, gk))
out.append(report("no start (5-20 s)", X_of(lo=5)))
out.append(report("first 10 s only", X_of(hi=10)))
busy = X_of(overlay)
out.append(report("busy: video overlay, retrained", busy))
out.append(report("busy: trained quiet, tested busy", base, Xte=busy))
for name, kw in [("rate 54 Mb/s", dict(rate=54e6)), ("rate 600 Mb/s", dict(rate=600e6)),
                 ("tau 0.1 ms", dict(tau=1e-4)), ("tau 5 ms", dict(tau=5e-3)), ("noise 0.1", dict(noise=.1))]:
    out.append(report(name, X_of(**kw)))
for ps in [50, 200, 1000]:
    d = X_of(fake(ps))
    out.append(report(f"defense {ps}/s: attacker not retrained", base, Xte=d))
    out.append(report(f"defense {ps}/s: attacker retrained", d))
(ROOT / "stress_results.json").write_text(json.dumps(out, indent=1))
