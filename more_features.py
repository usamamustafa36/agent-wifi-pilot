#!/usr/bin/env python3
"""Exploratory: does a larger feature set help? Not pre-registered.

Feature sets and models are compared by time-block cross-validation on session 1 only.
The pick is then scored on session 2, which has already been seen, so it is not fresh confirmation.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    """Whole-run shape, read from the signal only: when it is active and how bursts behave."""
    m = len(v) // 400                                  # 50 ms steps
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
    out += [np.searchsorted(cum, q) / FS for q in (.1, .25, .5, .75, .9)]        # when the energy arrives
    z = e - e.mean()
    spec = np.abs(np.fft.rfft(z)) ** 2
    f = np.fft.rfftfreq(len(z), .05)
    tot = spec[1:].sum() + 1e-9
    out += [spec[(f >= a) & (f < b)].sum() / tot for a, b in ((0, .25), (.25, .5), (.5, 1), (1, 2), (2, 4), (4, 10.01))]
    ac = [np.dot(z[:-l], z[l:]) / (np.dot(z, z) + 1e-9) for l in (1, 2, 5, 10, 20, 40)]
    pct = np.percentile(v, [5, 25, 50, 75, 90, 95, 99, 99.9])
    return np.r_[out, ac, pct, v.mean(), v.std(), ((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-9),
                 ((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-9)]


def feature_sets(v):
    b = bursts(v)
    seg12 = segs(v, 12)
    summ = np.r_[v.mean(), v.std(), np.percentile(v, [50, 90, 99])]
    return {"original 29": np.r_[seg12, summ], "burst shape": b, "fine 0.5 s": segs(v, 40),
            "fine + burst": np.r_[segs(v, 40), b], "all": np.r_[segs(v, 40), seg12, b]}


def build(runs, pk, salt, per_s=0):
    res = {}
    for r, (t, n) in zip(runs, pk):
        s = r["run"] + salt
        if per_s:
            g = np.random.default_rng(s + 7)
            k = g.poisson(per_s * W)
            t, n = np.r_[t, g.uniform(0, W, k)], np.r_[n, np.full(k, 1400)]
        for k_, x in feature_sets(signal(t, n, s)).items():
            res.setdefault(k_, []).append(x)
    return {k: np.stack(v) for k, v in res.items()}


MODELS = {"logistic C=0.3": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=4000)),
          "logistic C=0.03": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.03, max_iter=4000)),
          "forest": lambda: RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=-1)}


def cv(mk, X, y, k=5):
    hit = np.zeros(len(y), bool)
    for te in np.array_split(np.arange(len(y)), k):
        tr = np.setdiff1d(np.arange(len(y)), te)
        hit[te] = mk().fit(X[tr], y[tr]).predict(X[te]) == y[te]
    return float(hit.mean())


rtr, ptr, ytr = load(HERE / "retry")
rte, pte, yte = load(HERE / "session2")
Ctr, Cte = build(rtr, ptr, 0), build(rte, pte, 100000)
print({k: v.shape[1] for k, v in Ctr.items()}, flush=True)
table = []
for s in Ctr:
    for m, mk in MODELS.items():
        table.append(dict(features=s, model=m, cv_session1=cv(mk, Ctr[s], ytr)))
        print(f"{s:14s} {m:16s} session-1 CV {table[-1]['cv_session1']:.3f}", flush=True)
best = max(table, key=lambda r: r["cv_session1"])
s, mk = best["features"], MODELS[best["model"]]
print("\nPICKED ON SESSION 1:", best, flush=True)
g = np.random.default_rng(0)
clean = mk().fit(Ctr[s], ytr)
pred = clean.predict(Cte[s])
acc = float((pred == yte).mean())
null = [float((mk().fit(Ctr[s], g.permutation(ytr)).predict(Cte[s]) == yte).mean()) for _ in range(200)]
out = dict(table=table, best=best, session2=dict(acc=acc, shuffled=float(np.mean(null)), p=(1 + sum(v >= acc for v in null)) / 201))
print("session 2 clean", out["session2"], flush=True)
base = MODELS["logistic C=0.3"]
print("original 29 session 2 clean", float((base().fit(Ctr["original 29"], ytr).predict(Cte["original 29"]) == yte).mean()), flush=True)
idx = g.integers(0, len(yte), (5000, len(yte)))
out["defense"] = []
for ps in [50, 200, 1000]:
    Dtr, Dte = build(rtr, ptr, 0, ps), build(rte, pte, 100000, ps)
    h0 = clean.predict(Dte[s]) == yte
    h1 = mk().fit(Dtr[s], ytr).predict(Dte[s]) == yte
    lo, hi = np.percentile(h1[idx].mean(1) - h0[idx].mean(1), [2.5, 97.5])
    row = dict(per_s=ps, not_retrained=float(h0.mean()), retrained=float(h1.mean()), ci=[float(lo), float(hi)])
    out["defense"].append(row)
    print("defense", row, flush=True)
if best["model"] != "forest":
    print("(importance is printed for the forest only)")
f = RandomForestClassifier(500, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Ctr["burst shape"], ytr)
names = [f"{n}@{th}" for th in (.1, .2, .4) for n in ("first_active", "last_active", "active_frac", "n_bursts", "burst_mean", "burst_max", "gap_mean", "gap_max")] + \
        [f"energy_t{q}" for q in (10, 25, 50, 75, 90)] + [f"band_{b}" for b in ("0-.25", ".25-.5", ".5-1", "1-2", "2-4", "4-10")] + \
        [f"autocorr_{l}" for l in (1, 2, 5, 10, 20, 40)] + [f"pct{q}" for q in (5, 25, 50, 75, 90, 95, 99, 99.9)] + ["mean", "std", "skew", "kurt"]
o = np.argsort(-f.feature_importances_)[:10]
out["burst_top"] = [(names[i], round(float(f.feature_importances_[i]), 3)) for i in o]
print("burst-shape forest top:", out["burst_top"], "session 2 acc", float((f.predict(Cte["burst shape"]) == yte).mean()))
(HERE / "session2" / "more_features_results.json").write_text(json.dumps(out, indent=1))
