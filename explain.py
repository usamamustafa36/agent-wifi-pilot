#!/usr/bin/env python3
"""Exploratory explanation of the logistic-regression attacker. NOT part of the pre-registered test.

Train on retry/, score on session2/, same features as confirm.py. Session 2 has now been looked at
repeatedly, so these numbers describe the model and do not count as fresh confirmation.

  ablation      remove a feature group and retrain (drop in accuracy)
  permutation   shuffle a feature group across test runs, model unchanged (drop in accuracy)
  coefficients  read the linear model directly
  tree          depth-3 decision tree as a second glass box
  shift         how far the defense moves each feature, in clean training standard deviations
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from analyze import pulses, FS

HERE = Path(__file__).parent
CL = ["chat", "code", "web"]
W = 20
NAMES = [f"mean_{i*20/12:.1f}-{(i+1)*20/12:.1f}s" for i in range(12)] + [f"std_{i*20/12:.1f}-{(i+1)*20/12:.1f}s" for i in range(12)] + \
        ["overall_mean", "overall_std", "p50", "p90", "p99"]
GROUPS = {
    "early 0-6.7 s": [i for i in range(24) if i % 12 < 4],
    "middle 6.7-13.3 s": [i for i in range(24) if 4 <= i % 12 < 8],
    "late 13.3-20 s": [i for i in range(24) if i % 12 >= 8],
    "segment means (activity)": list(range(12)),
    "segment stds (burstiness)": list(range(12, 24)),
    "whole-run summaries": list(range(24, 29)),
}


def load(root):
    runs = sorted(map(json.loads, open(root / "runs.jsonl")), key=lambda r: r["start"])
    runs = [r for r in runs if r["cls"] in CL]
    pk = []
    for r in runs:
        d = [(float(x["t"]), int(x["len"])) for x in csv.DictReader(open(root / "data" / r["file"])) if x["dir"] == "down"]
        pk.append((np.array([a for a, _ in d]), np.array([b for _, b in d])))
    return runs, pk, np.array([CL.index(r["cls"]) for r in runs])


def feats(t, n, seed):
    x = pulses(t, n, W, 200e6)
    a = 1 / (1 + 1e-3 * FS)
    v = lfilter([a], [1, a - 1], x)
    v = v + np.random.default_rng(seed).normal(0, .02, v.shape)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    seg = np.array_split(v, 12)
    return np.r_[[s.mean() for s in seg], [s.std() for s in seg], v.mean(), v.std(), np.percentile(v, [50, 90, 99])]


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


def explain(name, Xtr, ytr, Xte, yte):
    m = mk().fit(Xtr, ytr)
    base = (m.predict(Xte) == yte).mean()
    g = np.random.default_rng(0)
    idx = g.integers(0, len(yte), (2000, len(yte)))
    rows = {}
    for gname, ix in GROUPS.items():
        keep = np.setdiff1d(np.arange(Xtr.shape[1]), ix)
        hab = mk().fit(Xtr[:, keep], ytr).predict(Xte[:, keep]) == yte
        h0 = m.predict(Xte) == yte
        d = h0[idx].mean(1) - hab[idx].mean(1)
        perm = []
        for _ in range(200):
            P = Xte.copy(); P[:, ix] = P[g.permutation(len(yte))][:, ix]
            perm.append(base - (m.predict(P) == yte).mean())
        rows[gname] = dict(ablation_drop=float(h0.mean() - hab.mean()), ablation_ci=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                           permutation_drop=float(np.mean(perm)))
    coef = m[-1].coef_
    top = {CL[c]: [(NAMES[i], round(float(coef[c, i]), 2)) for i in np.argsort(-coef[c])[:4]] for c in range(3)}
    print(f"\n== {name}  accuracy {base:.3f}", flush=True)
    for k, v in rows.items():
        print(f"  {k:28s} ablation {v['ablation_drop']:+.3f} CI [{v['ablation_ci'][0]:+.3f},{v['ablation_ci'][1]:+.3f}]  permutation {v['permutation_drop']:+.3f}")
    print("  strongest positive weights:", top)
    return dict(acc=float(base), groups=rows, top=top), coef


rtr, ptr, ytr = load(HERE / "retry")
rte, pte, yte = load(HERE / "session2")
Ctr, Cte = build(rtr, ptr, 0), build(rte, pte, 100000)
out, coefs = {}, {}
out["clean"], coefs[0] = explain("clean attacker on clean test", Ctr, ytr, Cte, yte)
sc = StandardScaler().fit(Ctr)
for ps in [50, 200]:
    Dtr, Dte = build(rtr, ptr, 0, ps), build(rte, pte, 100000, ps)
    out[f"retrained_{ps}"], coefs[ps] = explain(f"retrained attacker, {ps}/s", Dtr, ytr, Dte, yte)
    shift = (sc.transform(Dte) - sc.transform(Cte)).mean(0)
    o = np.argsort(-np.abs(shift))[:6]
    group_shift = {k: round(float(np.abs(shift[v]).mean()), 2) for k, v in GROUPS.items()}
    cos = float((coefs[0] * coefs[ps]).sum() / (np.linalg.norm(coefs[0]) * np.linalg.norm(coefs[ps])))
    out[f"shift_{ps}"] = dict(groups=group_shift, top=[(NAMES[i], round(float(shift[i]), 2)) for i in o], weight_cosine_with_clean=cos)
    print(f"  defense {ps}/s mean shift (SDs):", group_shift, "\n  biggest:", out[f'shift_{ps}']['top'], "\n  weight cosine vs clean model:", round(cos, 3), flush=True)
t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(Ctr, ytr)
out["tree_acc"] = float((t.predict(Cte) == yte).mean())
print("\n== depth-3 tree accuracy", round(out["tree_acc"], 3))
print(export_text(t, feature_names=NAMES, class_names=CL))
(HERE / "session2" / "explain_results.json").write_text(json.dumps(out, indent=1))
