# Confirmation session: fixed before recording

Written before any session-2 data exists. Script: `confirm.py`.

**Data.** Training: `retry/` chat, code and web runs only (90 runs). Testing: `session2/`, 30 new runs per
class, 20 s window, Tor off, same model (Haiku), new prompt wording, new topics and coding tasks.

**Model.** The stress-test logistic regression and features, unchanged. Default sniffer simulation:
200 Mb/s, 1 ms smoothing, noise 0.02. No tuning on session 2. The LSTM is not used here.

**Primary test.** Three-way accuracy on session 2 with the simulated sniffer. Pass means the accuracy beats
1000 shuffled-training-label runs at p < 0.05. Chance is 33%.

**Secondary tests, defense.** Fake traffic at 50, 200 and 1000 packets/s added to the test runs. Attacker not
retrained: trained on clean session 1. Attacker retrained: trained on session 1 with the same defense level.
The defense claim is supported at a level only if retrained beats not retrained, with the paired bootstrap
95% interval above zero, and retrained also beats shuffled labels at p < 0.05. Three levels are tested,
so each result is also reported against a Bonferroni threshold of p < 0.0167.

**Also reported.** Real-packet accuracy, the confusion matrix, and whether session 2 ran on a different day or
network. All outcomes are reported, including failure.

**Known limits.** The sniffer is simulated. Session 2 may be on the same day and network as session 1.
One tool and one model.
