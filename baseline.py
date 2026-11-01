#!/usr/bin/env python3
"""Follow-up after the LSTM scored at chance: is the failure the signal or the model?

Added AFTER seeing the LSTM result, so this is exploratory, not a pre-planned test.
Simple classifier, contiguous time-block folds, label-permutation p-value.
Views: real packet counts/bytes (no sniffer model) and the simulated trace.
"""
import csv, json, os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyze import trace, STEPS

ROOT = Path(os.environ.get("PILOT_DIR", Path(__file__).parent))
runs = sorted((r for r in map(json.loads, open(ROOT / "runs.jsonl")) if r["run"] < 9000), key=lambda r: r["start"])


def load(r):
    rows = list(csv.DictReader(open(ROOT / "data" / r["file"])))
    d = [(float(x["t"]), int(x["len"])) for x in rows if x["dir"] == "down"]
    return np.array([a for a, _ in d]), np.array([b for _, b in d])


def bins(t, n, edges):
    return np.histogram(t, edges, weights=n)[0].astype(float)


def raw_features(t, n):
    b = bins(t, n, np.arange(0, 21, 2))
    return np.log1p(np.r_[b, b.sum(), len(n), np.percentile(n, [50, 90]) if len(n) else [0, 0]])


def sim_features(t, n, r):
    x = trace(t, n, r["window"], 200e6, 1e-3, 0.02, np.random.default_rng(r["run"]))
    k = STEPS // 12
    seg = x[:k * 12].reshape(12, k)
    return np.r_[seg.mean(1), seg.std(1), x.mean(), x.std(), np.percentile(x, [50, 90, 99])]


def cv(X, y, folds=4):
    hit = np.zeros(len(y), bool)
    for te in np.array_split(np.arange(len(y)), folds):
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X[tr], y[tr])
        hit[te] = m.predict(X[te]) == y[te]
    return hit.mean()


def report(name, X, y, perms=500):
    acc = cv(X, y)
    rng = np.random.default_rng(0)
    null = [cv(X, rng.permutation(y)) for _ in range(perms)]
    p = (1 + sum(v >= acc for v in null)) / (perms + 1)
    print(f"{name:34s} n={len(y)} acc={acc:.3f} shuffled_mean={np.mean(null):.3f} p={p:.3f}", flush=True)
    return dict(name=name, n=len(y), acc=float(acc), shuffled_mean=float(np.mean(null)), p=float(p))


data = [(r, *load(r)) for r in runs]
out = []
for label, cl in [("agent tasks", ["chat", "code", "web"]), ("non-AI control", ["video", "browse", "download", "idle"]),
                  ("all seven", ["chat", "code", "web", "video", "browse", "download", "idle"])]:
    d = [x for x in data if x[0]["cls"] in cl]
    y = np.array([cl.index(x[0]["cls"]) for x in d])
    out.append(report(label + " | real packets", np.stack([raw_features(t, n) for _, t, n in d]), y))
    out.append(report(label + " | simulated trace", np.stack([sim_features(t, n, r) for r, t, n in d]), y))
d = [x for x in data if x[0]["cls"] in ("chat", "code", "web", "idle")]
y = np.array([int(x[0]["cls"] != "idle") for x in d])
out.append(report("agent vs idle | real packets", np.stack([raw_features(t, n) for _, t, n in d]), y))
out.append(report("agent vs idle | simulated trace", np.stack([sim_features(t, n, r) for r, t, n in d]), y))
(ROOT / "baseline_results.json").write_text(json.dumps(out, indent=1))
