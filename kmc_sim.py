"""Toy test of the KMC averaging claim. Assumed values, not hardware.
Field at sniffer: h(t) = 1 + body(t) + ris(t). Power V = |h|^2 + receiver noise.
Attacker knows no key: plain low-pass, then picks the strongest 0.1-0.5 Hz peak."""
import numpy as np

FS, T = 200, 60
t = np.arange(FS * T) / FS


def trial(ris_amp, hold_s, body=0.05, noise=0.02, rng=None):
    f0 = rng.uniform(0.15, 0.45)
    b = body * np.exp(1j * rng.uniform(0, 6.28)) * np.sin(2 * np.pi * f0 * t)
    n = max(1, int(hold_s * FS))
    idx = np.arange(len(t)) // n
    phases = rng.uniform(0, 2 * np.pi, idx[-1] + 1)      # keyed PRNG looks the same to a keyless attacker
    h = 1 + b + ris_amp * np.exp(1j * phases[idx])
    v = np.abs(h) ** 2 + rng.normal(0, noise, len(t))
    win = np.hanning(len(v))
    spec = np.abs(np.fft.rfft((v - v.mean()) * win))
    f = np.fft.rfftfreq(len(v), 1 / FS)
    band = (f >= 0.1) & (f <= 0.5)
    return abs(f[band][spec[band].argmax()] - f0) < 0.03


rng = np.random.default_rng(7)
print("Breathing-rate attack success (error < 0.03 Hz, 300 trials per cell). Chance is about 15%.")
print(f"{'RIS amplitude vs direct':>25} | " + " | ".join(f"{n:>11}" for n in ["5 ms hold", "50 ms", "500 ms", "2 s", "5 s"]))
for amp in (0.0, 0.1, 0.3, 1.0):
    row = [np.mean([trial(amp, h, rng=rng) for _ in range(300)]) for h in (0.005, 0.05, 0.5, 2, 5)]
    print(f"{amp:>25} | " + " | ".join(f"{x*100:>10.0f}%" for x in row))
