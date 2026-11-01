# Hide-the-end defense test: fixed before running

Attackers (fixed): logistic regression on the 29-feature and 133-feature sets, trained on session 1
and tested on session 2. The defenses are applied to recorded packets; nothing is new on the air.

Session 2 has been used to evaluate attacks, but not these defenses. Results are exploratory until
repeated on a new session.

**Defenses**
- `tail_steady`: after the agent exits, send 50 packets/s of 1400 bytes until 20 s.
- `tail_replay`: after the agent exits, replay a random same-session web run's packets from that time to 20 s.
- `shape_R`: send all traffic in fixed 1400-byte slots at R = 100 and 200 kB/s, using dummies when idle and
  queueing real data. Latency is the queue delay, reported as median and maximum.
- `random_same_budget`: random 1400-byte packets over the full 20 s, at the same mean extra bytes as each
  tail defense. This is the comparison.

**Pass rule.** A defense holds if the retrained attacker does not beat shuffled labels at p < 0.05 for that
attacker (200 shuffles). A tail defense counts as better than random only if retrained accuracy is lower,
with the paired bootstrap 95% interval below zero, for the same attacker and budget.

**Reported for all:** not-retrained and retrained accuracy, extra MB per 20 s, latency. Failures are reported too.

**Known limit:** padding to a fixed 20 s says nothing about tasks longer than 20 s.
