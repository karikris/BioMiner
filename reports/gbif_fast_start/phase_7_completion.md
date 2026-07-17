# Phase 7 completion — class-specific reference-bank evaluation

Phase 7 is complete. All four task commits were pushed and the final full
repository run passed 2,428 tests in 87.18 seconds.

| Task | Commit | Result |
|---|---|---|
| 7.1 | `d71bceabf75748a25df39d0025e8da907f295f8c` | Typed audit, summary, policy and report contracts across eleven required dimensions |
| 7.2 | `b434f4df9707a16469b7371a1eea51ae65ac74fb` | Weighted human-audit metrics, confidence intervals and explicit unavailable states |
| 7.3 | `7111f664e1edc4d967d069390dd8ec40454288fe` | Error groups related to descriptive reference-quality evidence without causal identity claims |
| 7.4 | `843aeb3dc604ee28c4d1d260447e85a1a451f3b2` | Versioned per-species escalation with every reason and threshold persisted |

The audit contract groups human-reviewed Flickr evidence by target species,
competitor, region, route, life stage, visual domain, source dataset, admission
basis, verification basis, sampling campaign and sampling stratum. Targeted
campaign rows require positive inclusion probabilities, and their metrics use
inverse-probability weights. Empty or underpowered groups remain explicitly
unavailable or insufficient and cannot be quality approval evidence.

For sufficiently reviewed groups, the evaluator reports precision, recall,
false-positive and false-negative rates, PR AUC, coverage, abstention,
competitor confusion and confidence intervals. Calibration metrics are emitted
only when every contributing row contains an attested calibrated probability;
otherwise the output reports raw-margin quantiles without relabelling margins
as probabilities.

Only explicitly selected underperforming species are joined to reference
dispersion, outlier, influence, route, provider, geographic, competitor and
observer evidence. These associations prioritize review. They are never proof
that any reference image has the wrong identity.

The escalation policy flags a species-reference group when a configured
performance bound, error rate, sample-size, dispersion, outlier, route-balance
or support threshold is breached. Each flag stores the reason, observed value,
comparison operator, policy threshold, policy fingerprint and decision
fingerprint. The resulting action is targeted human review; the statistical
identity conclusion remains `not_assessed`.

These are fixture-tested software and statistical contracts, not results from a
live reference-bank audit. No production accuracy, calibration, class-confusion,
reviewer-throughput, cost or time-saving improvement is claimed.
