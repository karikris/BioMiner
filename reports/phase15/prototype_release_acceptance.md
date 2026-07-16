# Phase 15 prototype release acceptance

Status: **accepted for the Build Week prototype only, with limitations**.

The `Papilio demoleus` target-aware prototype satisfies all architecture, data,
operations, evaluation, and 17 completion checks. It is not authorized as a
scientific release, a production-default migration, or a public reference-image
product. This verification used local storage only; S3 was neither permitted nor
accessed.

## Frozen identity

| Item | Value |
|---|---|
| Classification mode | `build_week_target_aware_prototype` |
| BioCLIP | `hf-hub:imageomics/bioclip-2.5-vit-huge-patch14-378` |
| Model revision | `191d741545e4c741cdef4b22c6eb69c945c1e592` |
| YOLOE checkpoint | `yoloe-26s-seg.pt` |
| Reference bank | `papilio-demoleus-prototype-bank-20260716` |
| Reference-bank SHA | `sha256:a5a26fc7e60c79dbcca9b9f8fc1ba2b3558e6307bd7c5b70711fe1ad022fde10` |
| Reference planner | `trust-first-layered-regional-v1.0.0` |
| Classifier fingerprint | `sha256:92638010837bd31b1cdbfae366fa1245015153b34f58dc8e8fea0a6378ad4acd` |
| Calibrator | Not fitted; no independently reviewed calibration labels |
| Margin policy | `prototype-raw-margin-abstention-v1.0.0`, threshold `0.10` |

## Acceptance summary

| Area | Result | Key evidence |
|---|---|---|
| Architecture | PASS | Target and complete candidate union always scored; no hierarchy pruning or spatial crop; raw full-frame BioCLIP; YOLOE router only |
| Data | PASS | Provenance, licence, attribution, trust, geography and route retained; exact duplicates removed; human verified remains `0` |
| Operations | PASS | Fail-closed readiness, persistent model reuse, content hashes, SQLite resume, no committed images/secrets/models |
| Evaluation | PASS | B0–B16 comparison, explicit margins/abstention, unlabelled distributions, failures, throughput and RSS memory |
| Completion criteria | 17/17 | Every task has a separate commit and `main` was pushed |

## Reference bank

The frozen bank has 81 provider-supported records and zero independently human
verified records. Trust is R4 for all 81. Geographic layers are A 51, B 6,
C 0, D 24, E 0. Licences are 2 allowed and 79 research-only. Routes are
80 adult, 1 larval and 0 pinned specimen.

Provider support is not represented as human verification. Flickr query matches
are not labels. Model output is screening evidence, not taxonomic validation.

## Operational result

The staged workload planned 13,501 records, classified 13,496, and retained five
download/decode failures as retryable. Target scoreability was 13,496/13,496.
P1 classified 100 records at 2.012120 records/s; P2 classified 1,000 at
2.391706 records/s; P3 classified 13,496 at 2.274524 records/s with peak RSS
1,765,261,312 bytes. MPS allocated, driver, and recommended-memory metrics remain
not instrumented.

The completed checkpoint resumed without new stage or model work. A fresh
five-image MPS smoke also passed using local temporary output.

## Prominent limitations

- No reference label has independent human taxonomic verification, so accuracy is not reported.
- The policy is uncalibrated and emits no probabilities.
- The staged runner’s diagnostic `0.02` margin abstention differs from the selected integration threshold `0.10`.
- 79 references are research-only; dashboards expose safe identifiers, not copies or URLs.
- Larval support is sparse; pinned-specimen and frozen visual-domain-negative support are absent.
- Two records lack owner evidence, so leakage protection is not claimed complete.
- Five Flickr and ten reference-source failures remain retryable.
- B11/B12 focused and masked inputs were unavailable.
- YOLOE routing accuracy is not independently validated.
- Inference distributions are not accuracy or prevalence estimates.

The machine-readable evidence and per-check limitations are in
`reports/phase15/prototype_release_acceptance.json`.
