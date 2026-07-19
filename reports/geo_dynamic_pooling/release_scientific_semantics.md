# Geography-conditioned dynamic pooling — scientific-semantics gate

Verdict: **all ten required software semantics pass** at workflow SHA
`670185286ab78f4d538cfd2bc222fef1e7d8da7e`.

The focused gate ran 21 executable behavior/policy checks in 0.34 seconds. The
GBIF matrix contributes 11 parameterized policy cases; the remaining checks
exercise the production candidate, scoring, review, quality, outcome and
handoff contracts directly.

| Required semantic | Executable result |
|---|---|
| Family cannot catastrophically prune | A wrong-family target and the complete candidate union survive family-first-safe scheduling. |
| Geography cannot certify identity | No-geo stays explicit and unavailable; the accepted contract says geography is evidence, never proof or absence. |
| GBIF references remain provisional | Every automatic matrix case has `human_verified = false`; human override remains separate. |
| Embeddings are not recomputed per pool | Cached-vector scoring reports zero encoder invocations, zero image materializations and one cached query vector consumed. |
| Raw evidence is not probability | Raw components, fusion evidence, calibrated probability and release authority remain separate fields. |
| Statistically supported is not human verified | Population support creates no item review event, release permit or publication authority. |
| Unreviewed cannot enter occurrence export | Release, screening and unresolved unreviewed lanes all fail the verified exporter. |
| Insufficient strata remain unavailable | Quality estimates are null with explicit insufficient-sample reasons, not zero. |
| Targeted samples do not support unweighted claims | Inclusion probability and sampling weight are null; representative eligibility and release authority are false. |
| Downstream maturity is preserved | TaxaLens quality remains unavailable rather than zero; ButterflyLens retains database, reviewer and release authority. |

## Authority boundary

This gate verifies software semantics. It does not supply live source evidence,
complete source-bound human review, produce statistical support, select a
candidate strategy/pool/fusion default, or authorize occurrence release. The
Phase 15 decision remains `insufficient_evidence`, with zero eligible variants
and unchanged runtime settings.

No GitHits call was made, per the user's directive. The subtask provenance has
`solution_id: null`; GitHits contributed no code or architecture.
