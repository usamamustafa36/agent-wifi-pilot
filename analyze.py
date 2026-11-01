#!/usr/bin/env python3
"""Turn recorded packets into a SIMULATED harvester trace and test four conditions.

The sniffer is simulated: real packet times and sizes, assumed physics.
  packet -> pulse (airtime at an assumed PHY rate) -> RC smoothing -> noise
  -> min-max normalisation (MetaSpy section 4) -> averaged to a short sequence.

Conditions, all scored on held-out runs (contiguous time blocks):
  clean          trained on clean, tested on clean
  defended       trained on clean, tested with random fake pulses added
  adaptive       trained on defended, tested on defended (attacker retrains)
  shuffled       clean, with training labels shuffled (should sit at chance)
"""
import argparse, csv, json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path(os.environ.get("PILOT_DIR", Path(__file__).parent))
FS = 20_000          # MetaSpy's sampling rate
STEPS = 1200         # 50 ms per step over a 60 s window
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def pulses(times, lens, window, rate_bps, overhead_s=100e-6):
    x = np.zeros(int(window * FS) + 1, dtype=np.float32)
    for t, n in zip(times, lens):
        a = int(t * FS)
        d = max(1, int((n * 8 / rate_bps + overhead_s) * FS))
        x[a:a + d] = 1.0
    return x


def fake_traffic(window, per_s, rng, size=1400):
    k = rng.poisson(per_s * window)
    return np.sort(rng.uniform(0, window, k)), np.full(k, size)


def trace(times, lens, window, rate, tau, noise, rng):
    x = pulses(times, lens, window, rate)
    a = 1 / (1 + tau * FS)
    y = np.empty_like(x); s = 0.0
    for i, v in enumerate(x):
        s += a * (v - s); y[i] = s
    y += rng.normal(0, noise, y.shape).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    n = len(y) // STEPS
    return y[:n * STEPS].reshape(STEPS, n).mean(1)


class Net(nn.Module):
    def __init__(s, k):
        super().__init__()
        s.lstm = nn.LSTM(1, 64, 2, batch_first=True, bidirectional=True, dropout=.2)
        s.head = nn.Linear(128, k)

    def forward(s, x):
        o, _ = s.lstm(x[..., None])
        return s.head(o.mean(1))


def fit_predict(Xtr, ytr, Xte, k, seed, epochs):
    torch.manual_seed(seed)
    net = Net(k).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xtr_t), device=DEV).split(16):
            opt.zero_grad()
            nn.functional.cross_entropy(net(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, device=DEV)).argmax(1).cpu().numpy()


def run(classes, a, runs):
    keep = [r for r in runs if r["cls"] in classes]
    keep.sort(key=lambda r: r["start"])
    y = np.array([classes.index(r["cls"]) for r in keep])
    clean, defended = [], []
    for r in keep:
        rows = list(csv.DictReader(open(ROOT / "data" / r["file"])))
        down = [(float(x["t"]), int(x["len"])) for x in rows if x["dir"] == "down"]
        t = np.array([d[0] for d in down]); n = np.array([d[1] for d in down])
        rng = np.random.default_rng(r["run"])
        clean.append(trace(t, n, r["window"], a.rate, a.tau, a.noise, rng))
        ft, fn = fake_traffic(r["window"], a.fake_per_s, rng)
        defended.append(trace(np.r_[t, ft], np.r_[n, fn], r["window"], a.rate, a.tau, a.noise, rng))
    clean, defended = np.stack(clean), np.stack(defended)
    folds = np.array_split(np.arange(len(y)), a.folds)  # contiguous, so train/test are different sittings
    res = {c: [] for c in ["clean", "defended", "adaptive", "shuffled"]}
    for s in range(a.seeds):
        rng = np.random.default_rng(1000 + s)
        hits = {c: np.zeros(len(y), bool) for c in res}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            m, sd = clean[tr].mean(), clean[tr].std() + 1e-9
            z = lambda v: (v - m) / sd
            p = fit_predict(z(clean[tr]), y[tr], z(np.stack([clean[te], defended[te]]).reshape(-1, STEPS)),
                            len(classes), s, a.epochs)
            hits["clean"][te] = p[:len(te)] == y[te]
            hits["defended"][te] = p[len(te):] == y[te]
            hits["adaptive"][te] = fit_predict(z(defended[tr]), y[tr], z(defended[te]), len(classes), s, a.epochs) == y[te]
            hits["shuffled"][te] = fit_predict(z(clean[tr]), rng.permutation(y[tr]), z(clean[te]), len(classes), s, a.epochs) == y[te]
        for c in res:
            res[c].append(float(hits[c].mean()))
    return dict(classes=classes, n=len(y), chance=float(np.bincount(y).max() / len(y)),
                **{c: dict(mean=float(np.mean(v)), seeds=v) for c, v in res.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=200e6, help="assumed PHY rate")
    p.add_argument("--tau", type=float, default=1e-3, help="assumed harvester smoothing")
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--fake-per-s", type=float, default=50)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    runs = [json.loads(l) for l in open(ROOT / "runs.jsonl")]
    runs = [r for r in runs if r["run"] < 9000]  # smoke-test runs excluded
    groups = {"agent_tasks": ["chat", "code", "web"],
              "non_ai_control": ["video", "browse", "download", "idle"],
              "all_seven": ["chat", "code", "web", "video", "browse", "download", "idle"]}
    out = {"settings": vars(a), "device": DEV, **{k: run(v, a, runs) for k, v in groups.items()}}
    (ROOT / "results.json").write_text(json.dumps(out, indent=1))
    for key in groups:
        r = out[key]
        print(key, "n=", r["n"], "chance=%.3f" % r["chance"],
              {c: round(r[c]["mean"], 3) for c in ["clean", "defended", "adaptive", "shuffled"]})


if __name__ == "__main__":
    main()
