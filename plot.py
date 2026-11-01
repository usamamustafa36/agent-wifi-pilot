#!/usr/bin/env python3
"""Email figure, built from session2/confirm_results.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
r = json.load(open(HERE / "session2" / "confirm_results.json"))
d = r["defense"]
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
labels = ["No defense"] + [f"{x['per_s']} fake\npackets/s" for x in d]
w = .36
fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)
ax.bar(0, r["primary"]["acc"] * 100, w, color=BLUE)
ax.text(0, r["primary"]["acc"] * 100 + 1.5, f"{r['primary']['acc']*100:.0f}%", ha="center", fontsize=9, color=INK)
for i, x in enumerate(d, 1):
    a, b = x["not_retrained"] * 100, x["retrained"] * 100
    ax.bar(i - w / 2 - .01, a, w, color=BLUE, label="Attacker not retrained" if i == 1 else None)
    ax.bar(i + w / 2 + .01, b, w, color=ORANGE, label="Attacker retrained on the defense" if i == 1 else None)
    ax.text(i - w / 2 - .01, a + 1.5, f"{a:.0f}%", ha="center", fontsize=9, color=INK)
    ax.text(i + w / 2 + .01, b + 1.5, f"{b:.0f}%", ha="center", fontsize=9, color=INK)
ax.axhline(100 / 3, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
ax.text(0.41, 100 / 3 + 1.2, "chance 33%", ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks(range(4), labels, fontsize=9.5, color=INK)
ax.set_ylim(0, 100); ax.set_xlim(-.5, 3.65)
ax.set_ylabel("Accuracy on chat / code / web (%)", fontsize=9.5, color=MUTED)
ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0); ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color="#e6e5e0", lw=.7); ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c2")
ax.set_title("Light padding looks effective until the attacker retrains", loc="left", fontsize=12, color=INK, fontweight="bold", pad=24)
ax.text(0, 1.02, f"Simulated harvester from real packet traces. Trained on one session, tested on a new one (n={r['test_runs']}).",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
fig.tight_layout()
fig.savefig(HERE / "defense_pilot.png", facecolor=SURF)
