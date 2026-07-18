# Human decision: geography-conditioned dynamic-pooling policy

- Status: accepted
- Decision ID: `geography-conditioned-dynamic-pooling-policy-v1`
- Decision date: 2026-07-18
- Human decision owner: Kris Kari
- Production selection outcome: `insufficient_evidence`
- Runtime settings changed: no

## Decision

BioMiner accepts adaptive, geography-conditioned global/local reference pooling
as the target-aware production architecture. Family and geography may order or
expand a complete candidate union, but neither may delete the target, establish
species identity, or turn missing source evidence into biological absence. The
canonical model input is the full frame, and each compatible media/model/input
identity is embedded once and reused across candidate schedules and pool plans.

Global and local reference evidence remains explicit. Every candidate receives
global evidence; local evidence is either present or carries an exact
unavailable reason. Raw family, global, local, nearest-reference, top-k,
coverage, disagreement and rank-movement values are model evidence, not
probabilities. Calibration, human verification, statistical support,
release-readiness and downstream publication remain separate authorities.

The Phase 15 production-selection policy has been applied without weakening
its gates. All 24 fixture variants were evaluated against nine criteria. Zero
variants are eligible, so no candidate strategy, pool variant, fusion method or
selection-evidence fingerprint is selected. Runtime settings remain unchanged
and all selection fields remain null. The fixture projection is not a human
choice of a production default.

## Evidence binding

- Current and resulting settings fingerprint:
  `sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d`
- Safe reference-pool policy fingerprint:
  `sha256:08a5983f4e3c9d92894b5bcca2fbb18dd7a6d74114fdc90523ad29fde654cdc5`
- Production decision fingerprint:
  `sha256:43d034983485e789b8fa7c0428131f13c826d695781a28130b011e13b3bf3fb2`
- Complete 24-row selection-table fingerprint:
  `sha256:7368a623b9fbd9a665e2ae135c7da24c2adf9ed3a87b8e36241a3a2f14a676ec`
- Integrated pilot-report fingerprint:
  `sha256:ade039c9914c6fc720773eee7fbfb2141ff087f3abf869d9ab56b5f54dfa5d09`

The authoritative structured decision is
`reports/geo_dynamic_pooling/pilot/production_default_decision.json`. Its
Markdown rendering and the integrated pilot report are review aids; they do
not supersede the structured artifact or its exact fingerprints.

## Remaining authority gates

A later production-default selection requires all frozen gates to pass
together, including eligible source-bound human review, the 86-effective-review
minimum, required 30-independent-record subgroup floors, a reviewed-precision
lower bound of at least 0.95, comparable instrumented computation, and MPS peak
memory at or below 536,870,912 bytes. A later decision must bind the selected
settings to those exact evidence fingerprints; it may not silently reinterpret
this fixture pilot.

Unreviewed Flickr remains candidate evidence and cannot enter an occurrence
release. Planned representative work is not completed review, targeted failure
discovery cannot support unweighted population claims, and downstream import
does not confer release or publication maturity. Release continues to fail
closed.

## Explicit non-decisions

This decision does not claim:

- live dynamic-pool execution or measured biological accuracy;
- calibrated probabilities, sufficient statistical support or subgroup
  quality;
- superiority or measured rejection of any strategy, pool or fusion method;
- completed human review or a release-ready Flickr occurrence;
- that geography proves identity or that no-geography means absence; or
- that a TaxaLens or ButterflyLens handoff bypasses its consumer's database,
  review, rights, RLS, release or publication rules.
