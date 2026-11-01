#!/usr/bin/env python3
"""Test defenses that hide when a task ends. Plan: DEFENSE_TEST_PLAN.md."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyze import pulses, FS

HERE = Path(__file__).parent
CL = ["chat", "code", "web"]
W = 20
SIZE = 1400


def load(root):
    runs = sorted(map(json.loads, open(root / "runs.jsonl")), key=lambda r: r["start"])
    runs = [r for r in runs if r["cls"] in CL]
    pk = []
    for r in runs:
        d = [(float(x["t"]), int(x["len"])) for x in csv.DictReader(open(root / "data" / r["file"])) if x["dir"] == "down"]
        pk.append((np.array([a for a, _ in d]), np.array([b for _, b in d])))
    return runs, pk, np.array([CL.index(r["cls"]) for r in runs])


def signal(t, n, seed):
    x = pulses(t, n, W, 200e6)
    a = 1 / (1 + 1e-3 * FS)
    v = lfilter([a], [1, a - 1], x)
    v = v + np.random.default_rng(seed).normal(0, .02, v.shape)
    return ((v - v.min()) / (v.max() - v.min() + 1e-9)).astype(np.float32)


def segs(v, k):
    s = np.array_split(v, k)
    return np.r_[[a.mean() for a in s], [a.std() for a in s]]


def bursts(v):
    m = len(v) // 400
    e = v[:m * 400].reshape(400, m).max(1)
    out = []
    for th in (.1, .2, .4):
        on = e > th
        idx = np.flatnonzero(on)
        first, last = (idx[0], idx[-1]) if len(idx) else (0, 0)
        d = np.diff(np.r_[0, on.astype(int), 0])
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        dur = (en - st) * .05 if len(st) else np.zeros(1)
        gap = (st[1:] - en[:-1]) * .05 if len(st) > 1 else np.zeros(1)
        out += [first * .05, last * .05, on.mean(), len(st), dur.mean(), dur.max(), gap.mean(), gap.max()]
    cum = np.cumsum(v) / (v.sum() + 1e-9)
    out += [np.searchsorted(cum, q) / FS for q in (.1, .25, .5, .75, .9)]
    z = e - e.mean()
    spec = np.abs(np.fft.rfft(z)) ** 2
    f = np.fft.rfftfreq(len(z), .05)
    tot = spec[1:].sum() + 1e-9
    out += [spec[(f >= a) & (f < b)].sum() / tot for a, b in ((0, .25), (.25, .5), (.5, 1), (1, 2), (2, 4), (4, 10.01))]
    ac = [np.dot(z[:-l], z[l:]) / (np.dot(z, z) + 1e-9) for l in (1, 2, 5, 10, 20, 40)]
    pct = np.percentile(v, [5, 25, 50, 75, 90, 95, 99, 99.9])
    return np.r_[out, ac, pct, v.mean(), v.std(), ((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-9),
                 ((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-9)]


def features(v):
    return {"29": np.r_[segs(v, 12), v.mean(), v.std(), np.percentile(v, [50, 90, 99])],
            "133": np.r_[segs(v, 40), bursts(v)]}


def end_time(r):
    return r["agent_seconds"] if r["agent_seconds"] else W   # did not finish inside the window: nothing to hide


def steady(r, pk, runs, g):
    e = end_time(r)
    k = g.poisson(50 * max(0, W - e))
    return g.uniform(e, W, k), np.full(k, SIZE)


def replay(r, pk, runs, g):
    e = end_time(r)
    if e >= W:
        return np.zeros(0), np.zeros(0, int)
    donors = [i for i, x in enumerate(runs) if x["cls"] == "web"]
    t, n = pk[g.choice(donors)]
    keep = t >= e
    return t[keep], n[keep]


def apply_extra(fn, runs, pk, salt):
    out, extra = [], []
    for r, (t, n) in zip(runs, pk):
        g = np.random.default_rng(r["run"] + salt + 7)
        et, en = fn(r, pk, runs, g)
        extra.append(float(np.sum(en)))
        out.append(features(signal(np.r_[t, et], np.r_[n, en], r["run"] + salt)))
    return {k: np.stack([x[k] for x in out]) for k in out[0]}, float(np.mean(extra))


def random_budget(mb):
    per_s = mb / SIZE / W
    def f(r, pk, runs, g):
        k = g.poisson(per_s * W)
        return g.uniform(0, W, k), np.full(k, SIZE)
    return f


def shape(runs, pk, salt, rate_b):
    """Fixed slots: one 1400-byte frame each interval. Every class looks the same unless noise differs."""
    slots = np.arange(0, W, SIZE / rate_b)
    out, delays, extra, left = [], [], [], []
    for r, (t, n) in zip(runs, pk):
        q = np.argsort(t); t2, n2 = t[q], n[q]
        cap = np.arange(1, len(slots) + 1) * SIZE
        cum = np.cumsum(n2)
        k = np.minimum(np.searchsorted(cap, cum), len(slots) - 1)
        d = slots[k] - t2
        delays.extend(np.maximum(d, 0).tolist())
        left.append(float(max(0, cum[-1] - cap[-1])) if len(cum) else 0.)
        extra.append(float(len(slots) * SIZE - (n2.sum() if len(n2) else 0)))
        out.append(features(signal(slots, np.full(len(slots), SIZE), r["run"] + salt)))
    return {k: np.stack([x[k] for x in out]) for k in out[0]}, float(np.mean(extra)), \
        float(np.median(delays)), float(np.max(delays)), float(np.mean(left))


def mk():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=4000))


rtr, ptr, ytr = load(HERE / "retry")
rte, pte, yte = load(HERE / "session2")
Str = [features(signal(t, n, r["run"])) for r, (t, n) in zip(rtr, ptr)]
Ste = [features(signal(t, n, r["run"] + 100000)) for r, (t, n) in zip(rte, pte)]
Ctr = {k: np.stack([x[k] for x in Str]) for k in ("29", "133")}
Cte = {k: np.stack([x[k] for x in Ste]) for k in ("29", "133")}
clean = {k: mk().fit(Ctr[k], ytr) for k in Ctr}
clean_acc = {k: float((clean[k].predict(Cte[k]) == yte).mean()) for k in Ctr}
print("clean", clean_acc, flush=True)
results, HITS = [], {}


def score(name, Dtr, Dte, extra_mb, **meta):
    for k in ("29", "133"):
        h0 = clean[k].predict(Dte[k]) == yte
        m = mk().fit(Dtr[k], ytr)
        h1 = m.predict(Dte[k]) == yte
        g = np.random.default_rng(0)
        null = [float((mk().fit(Dtr[k], g.permutation(ytr)).predict(Dte[k]) == yte).mean()) for _ in range(200)]
        p = (1 + sum(v >= h1.mean() for v in null)) / 201
        HITS[(name, k)] = h1
        row = dict(defense=name, attacker=k, not_retrained=float(h0.mean()), retrained=float(h1.mean()),
                   shuffled=float(np.mean(null)), p=float(p), holds=bool(p >= .05), extra_mb=extra_mb, **meta)
        results.append(row)
        print(row, flush=True)


for name, fn in [("tail_steady", steady), ("tail_replay", replay)]:
    Dtr, btr = apply_extra(fn, rtr, ptr, 0)
    Dte, bte = apply_extra(fn, rte, pte, 100000)
    score(name, Dtr, Dte, bte / 1e6)
    Rtr, _ = apply_extra(random_budget(bte), rtr, ptr, 0)
    Rte, _ = apply_extra(random_budget(bte), rte, pte, 100000)
    rn = f"random_same_as_{name}"
    score(rn, Rtr, Rte, bte / 1e6)
    for k in ("29", "133"):
        idx = np.random.default_rng(1).integers(0, len(yte), (5000, len(yte)))
        d = HITS[(name, k)][idx].mean(1) - HITS[(rn, k)][idx].mean(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        row = dict(comparison=f"{name} minus random", attacker=k, diff=float(HITS[(name, k)].mean() - HITS[(rn, k)].mean()),
                   ci=[float(lo), float(hi)], better=bool(hi < 0))
        results.append(row)
        print(row, flush=True)
for kb in (100, 200):
    Dtr, _, _, _, _ = shape(rtr, ptr, 0, kb * 1000)
    Dte, ex, med, mx, left = shape(rte, pte, 100000, kb * 1000)
    score(f"shape_{kb}kBps", Dtr, Dte, ex / 1e6, median_delay_s=med, max_delay_s=mx, unsent_mb=left / 1e6)
(HERE / "session2" / "hide_end_results.json").write_text(json.dumps(dict(clean=clean_acc, results=results), indent=1))
