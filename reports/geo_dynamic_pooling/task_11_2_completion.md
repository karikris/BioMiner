# Task 11.2 completion — fail-closed remediation triggers and review queues

Status: completed and pushed to `origin/main` through
`b67fae6f71fabe7290db97d8ff97cf97b66ec1da`.

## Delivered

Versioned escalation decisions now map complete audited precision, routing,
disagreement, local-support, reference-influence, calibration and OOD failures
to typed human-review actions. Insufficient groups request evidence collection;
they are not labelled underperforming. Every rule records its source quality-row
fingerprint, comparison basis, observed value, operator and threshold.

The GBIF reference queue targets only candidates bound to a flagged family,
genus, species or geographic group. Overall/unbound reference triggers remain
explicitly unmatched. Priority is a transparent heuristic, identity remains
`not_assessed`, and targeting does not change support disposition.

The Flickr follow-up queue separates representative-expansion candidates from
targeted diagnostics. Representative candidates stop at a sampling-design gate:
no inclusion probability, design weight or estimation eligibility is invented.
Every row remains pending human review and outside occurrence release.

## Gate

- Escalation, new/legacy queue and authority suite: 79 passed in 2.34 seconds.
- Full regression: 3,071 passed in 117.36 seconds.
- Changed-file Ruff, format and `git diff --check`: passed.
- Remote `origin/main` resolved to `b67fae6…` after the task push.

## Claim boundary

A remediation flag is not a bad-reference or identity conclusion. Queue priority
is not probability, deterministic follow-up is not representative sampling, and
no queue grants release authority. Live audit evidence, sampling design and
human decisions remain required.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
