# ADR: statistical support and human-verification vocabulary

- Status: accepted
- Date: 2026-07-18
- Decision owner: Kris Kari
- Scope: BioMiner evidence labels, reports, exports and downstream handoffs
- Architecture dependency: `geography_conditioned_dynamic_pooling.md`

## Decision summary

BioMiner will represent model evidence, human review, calibration, statistical
support, release readiness and publication as separate, fingerprint-bound
states. None is a synonym for another, and no later-sounding label may be
inferred merely from an earlier state.

In particular:

- a reviewed item is not necessarily accepted;
- a calibrated score is not human verification;
- population- or stratum-level statistical support is not per-item review;
- a release-ready occurrence candidate is not yet a published occurrence; and
- a published occurrence requires a downstream publication receipt and does
  not retroactively turn its model or provider evidence into ground truth.

Machine scoring may run across the Flickr corpus before review, but the
existing source-bound human-review and release policy remains mandatory for a
final occurrence export unless a later explicit human decision changes it.

## Why this vocabulary is necessary

Dynamic global/local reference pools will produce more informative scores,
disagreement fields and review priorities. Those improvements increase the
risk that product language accidentally overstates the authority of model
evidence. Raw similarities are uncalibrated. Calibration estimates score
reliability for an applicable population. Statistical audit estimates
performance for a defined sampling frame. Human review records a person's
source-bound judgment. Release and publication add rights, taxonomy,
provenance and governance decisions.

One boolean such as `verified` cannot carry these distinctions. Every state
must identify its authority, subject, population, prerequisites, evidence
fingerprints, availability and limitations.

## Canonical states

### Model-scored candidate

**Meaning:** A versioned model/comparison plan produced candidate scores or an
abstention for an exact visual input.

**Required evidence:** source media hash, visual-input and route identity,
candidate and pool identities, model/revision/weights/preprocessing identity,
component scores, score policy, and producer commit.

**Allowed claims:** `model_scored_candidate`, `provisional_candidate`,
`review_priority_candidate`, or `failure_discovery_candidate`, as supported by
the exact policy.

**Forbidden claims:** probability, calibrated, human reviewed, human verified,
release ready, occurrence, or ground truth.

Provider-asserted GBIF support used by this stage remains
`GBIF provider-asserted provisional support`; it is not a calibration or
final-test label.

### Human-reviewed

**Meaning:** One or more authenticated, append-only human review events exist
for the exact source media and review question. The state describes process,
not outcome.

**Required evidence:** source hash, campaign/question/sampling identities,
reviewer group or privacy-preserving identity, assignment and blind-review
state, event order/time, effective-event rule, outcome, conflict/adjudication
state, and decision fingerprint.

**Allowed claims:** `human_reviewed` plus the exact effective outcome, such as
accepted, rejected, uncertain, conflict, insufficient detail, media failure or
skipped. A decisive positive taxon outcome may be described as
`human_supported` within its question scope.

**Forbidden claims:** every reviewed item is positive; every positive review
is expert-reviewed; review proves geographic occurrence; review alone is
release ready; or reviewer consensus makes model evidence ground truth.

`human_reviewed=false` means no qualifying review event under the specified
contract. An unavailable review store or unbound legacy annotation is
`unavailable` with a reason, not false.

### Calibrated

**Meaning:** A named calibrator transforms an eligible raw score into an
estimated probability for a precisely defined outcome and applicability
domain.

**Required evidence:** calibrator algorithm/version and fingerprint, training
and calibration split identities, source/owner/duplicate isolation, reviewed
label contract, target outcome, feature definition, applicability filters,
sample sizes, reliability metrics and intervals, and out-of-domain behavior.

**Allowed claims:** `calibrated_probability_available` for rows inside the
declared applicability domain, with the exact probability target (for example,
probability of a decisive positive review under the audited label contract).

**Forbidden claims:** raw cosine or fused score is a probability; calibration
proves identity; calibration replaces human review; or a calibrator remains
valid after model, pool, population, policy or label-contract drift.

Calibration availability is row-specific. Out-of-domain, stale or missing
calibration produces a null probability with an explicit reason.

### Statistically supported

**Meaning:** A preregistered or versioned statistical audit supports a stated
population- or stratum-level claim at its specified confidence/uncertainty
standard. It does not create a human review event for unsampled items.

**Required evidence:** sampling frame and snapshot, representative versus
targeted purpose, strata and clusters, inclusion probabilities, duplicate and
owner groups, reviewed outcomes, estimand, estimator, weighting, interval
method, effective sample size, coverage/selective-risk definition, thresholds,
missingness and insufficient-sample rules, and report fingerprint.

**Allowed item label:**
`statistically_supported_screening_candidate` only when the item satisfies the
audited selective policy and belongs to the population/stratum for which the
required support criterion passed.

**Allowed population claims:** the exact estimand and uncertainty statement,
for example a weighted precision lower bound for selected candidates in a
named stratum and snapshot.

**Forbidden claims:** statistically supported means human/community/expert
verified; an unsampled item was reviewed; a targeted failure-discovery sample
estimates population precision without a valid design; or a passing population
metric authorizes occurrence release.

An audit with too few independent outcomes, zero reviewed rows, invalid
weights, unresolved leakage or an unmet bound is `insufficient_sample` or
`not_supported`, never a guessed metric.

### Release-ready occurrence candidate

**Meaning:** One exact candidate has passed every applicable BioMiner release
gate and has an immutable evidence packet ready for an authorized downstream
release decision.

**Required evidence:** decisive source-bound human support under the release
policy, current taxonomy and source identity, coordinate precision and date,
duplicate/observation independence, rights and attribution, quality and expert
gates where required, no unresolved conflict, current model/review/policy
fingerprints, and a release-gate evidence ID.

**Allowed claim:** `release_ready_occurrence_candidate`.

**Forbidden claims:** already published; scientifically proven occurrence;
human review alone passed release; or BioMiner bypassed downstream authority.

Release fails closed. A missing, stale, withheld or not-applicable dependency
uses its exact availability state and reason; it is not coerced to pass.

### Published occurrence

**Meaning:** An authorized downstream system accepted a release-ready evidence
packet into a named occurrence product and returned an immutable publication
receipt.

**Required evidence:** release-ready packet ID and fingerprint, publisher and
product identity, authorized actor/role, publication policy/version,
publication timestamp and record identifier, rights/attribution state, source
and producer commits, and receipt fingerprint. Withdrawal/supersession state
must remain append-only.

**Allowed claim:** `published_occurrence` only within the named product and
receipt scope.

**Forbidden claims:** BioMiner model output alone is a publication; internal
export equals public publication; publication makes all upstream labels ground
truth; or one product's publication authorizes another product.

## Authority and implication matrix

| State | Subject | Authority | Can exist without human review? | Implies release ready? | Implies published? |
|---|---|---|---:|---:|---:|
| Model-scored candidate | Item | Model + plan artifact | Yes | No | No |
| Human-reviewed | Item/question | Effective review event(s) | No | No | No |
| Calibrated | Score within domain | Reviewed calibration audit | No for fitting; yes for an applicable scored item | No | No |
| Statistically supported | Population/stratum; conditional item screening label | Sampling and statistical audit | No for the audit; unsampled selected items remain unreviewed | No | No |
| Release-ready occurrence candidate | Item/evidence packet | Complete release policy | No | Yes, by definition | No |
| Published occurrence | Product record | Authorized downstream publication receipt | No | Requires it | Yes, in receipt scope |

The table defines only one-way prerequisites. For example, a published record
has a release-ready ancestor, but its existence does not imply that every
model score was calibrated or that every supporting reference image was human
verified.

## Required parallel outputs

BioMiner must preserve three non-overlapping decision destinations even when
they are delivered in one artifact family:

1. **Human-reviewed release set** — decisive, current, source-bound outcomes
   that also satisfy every applicable release gate.
2. **Statistically supported screening set** — unreviewed or reviewed candidates
   selected by an audited policy for screening, analysis or review planning;
   never an occurrence export by itself.
3. **Review-required/abstained set** — uncertainty, disagreement, unsupported
   strata, missing evidence, conflicts and policy abstentions with exact
   reasons and priorities.

An item may have evidence represented in multiple source tables, but one
release decision cannot simultaneously be `release_ready` and
`review_required`. Exports retain the evidence lineage that explains changes
over time rather than overwriting prior states.

## Statistical-support contract

### Sampling purpose

`representative_audit` estimates a declared population/stratum quantity and
requires known inclusion probabilities or a justified probability design.
`targeted_failure_discovery` and remediation queues find errors efficiently
but do not estimate corpus performance unless a valid estimator explicitly
supports their design. Purpose must be bound before outcomes are observed.

### Independence and leakage

Splits and variance calculations respect source owner, biological observation,
exact/perceptual duplicate, burst, media and acquisition groups. Calibration,
threshold selection, support construction and final audit/test data remain
isolated. Targeted follow-up derived from an outcome cannot be silently folded
into a representative estimate.

### Minimum report fields

Every statistical-support report includes:

- population, snapshot, stratum and selection-policy fingerprints;
- numerator/denominator definitions and all outcome counts;
- raw and weighted estimates where valid;
- cluster-aware uncertainty interval and effective sample size;
- selected coverage, abstention and review-required rates;
- probability/calibration availability and applicability;
- missing, withheld, unresolved and excluded counts with reasons;
- threshold-selection dataset and leakage checks; and
- status: `not_evaluated`, `insufficient_sample`, `not_supported`, or
  `supported`.

The word `supported` must be followed by the exact claim and criterion. A point
estimate without its design, denominator and uncertainty is not statistical
support.

## Calibration versus statistical support

Calibration and statistical support answer different questions:

- calibration asks whether an estimated probability aligns with the audited
  outcome frequency inside an applicability domain;
- statistical support asks whether evidence meets a declared population,
  stratum or selective-policy criterion with uncertainty accounted for.

A score can be calibrated but fail a precision/coverage support criterion. A
stratum can meet a weighted precision criterion while individual selected rows
have no calibrated probability. Reports therefore expose both states and do
not infer one from the other.

## Availability states and zero handling

All maturity fields use an explicit availability wrapper:

- `available`: value and supporting fingerprints are present;
- `unavailable`: evidence was not produced or cannot be accessed, with reason;
- `withheld`: evidence exists but policy/rights/privacy prevents disclosure,
  with reason;
- `not_applicable`: the contract does not apply, with reason.

For available count fields, zero is a real observed value. For unavailable
metrics, the value is null. Zero human reviews can be reported as a count of
zero, but precision, agreement and reliability remain unavailable or
insufficient—not zero.

## Permitted and prohibited language examples

| Evidence | Permitted | Prohibited |
|---|---|---|
| Raw dynamic-pool ranking | “Top provisional candidate; global/local raw score disagreement is 0.08.” | “92% confident occurrence.” |
| Positive review event | “Human-reviewed positive for the displayed question and source hash.” | “Published occurrence.” |
| Calibrated selective score | “Estimated positive-review probability 0.94 under calibrator X and applicability Y.” | “94% certain species truth.” |
| Passing representative audit | “The selected stratum meets criterion C under weighted estimate and interval I.” | “Every selected item is human verified.” |
| Complete release packet | “Release-ready occurrence candidate.” | “Already published.” |
| Publication receipt | “Published in product P as record R.” | “Ground truth for every future model and geography.” |

## Transition rules

1. Model scoring can create candidates, screening eligibility, abstention and
   review priority only.
2. Effective human events can create reviewed outcomes only for their bound
   media/question; adjudication rules resolve conflicts without deleting event
   history.
3. Calibration becomes available only through a reviewed, leakage-safe fit and
   validation artifact applicable to the current row.
4. Statistical support becomes available only through a valid sampling/audit
   report for the current population and policy.
5. Release readiness requires all gates independently; calibration or
   statistical support may be required by policy but cannot substitute for a
   human gate.
6. Publication requires release readiness plus downstream authorization and a
   receipt.
7. Drift, supersession, withdrawal, stale source bytes or policy changes append
   a new state and invalidate affected claims; historical evidence is never
   rewritten in place.

## Downstream mappings

TaxaLens may display geographic review/quality projections, but reviewed
positive and release-ready additional occurrence counts remain separate. Its
`reviewed-labels-v2` export supplies source-bound review evidence, not direct
publication authority.

ButterflyLens classification maturity maps BioMiner evidence into ordered
available/unavailable states for butterfly detected, species candidate,
community review, quality estimate, expert review and release ready.
BioMiner's handoff leaves `scientific_claim_allowed=false`; an RLS-protected
downstream adapter controls release transitions.

## Acceptance tests for all later phases

Later schemas, reports, CLI output and user-facing copy must demonstrate that:

1. no raw score or margin is named a probability;
2. reviewed process and reviewed outcome are separate;
3. calibration carries target, applicability and provenance;
4. statistical support names population, estimand, sampling design, interval
   and status;
5. zero-review states cannot emit precision or agreement as zero;
6. unreviewed statistically supported candidates remain screening-only;
7. every occurrence export requires current source-bound review and all release
   gates;
8. release readiness and publication use different identities and authorities;
9. unavailable/withheld/not-applicable evidence remains null with a reason; and
10. prohibited maturity language causes report/schema tests to fail.

## GitHits provenance

The required Subtask 0.2.2 GitHits search had an explicit 45-second bound and
returned no result. The unavailable attempt is recorded under
`geo-pool-0.2.2` in `provenance/githits.jsonl`; no solution ID, external
repository, code, prose or claimed precedent was invented. This decision
formalizes committed BioMiner, TaxaLens and ButterflyLens evidence boundaries.
