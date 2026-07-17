# Adaptive GBIF fast-start — final goal report

Implementation is complete at production-code SHA `477eaface3d1f5efa51255550f0ef8d6a7740f35` on `main`, with live scientific work explicitly pending. The final report commit is self-identifying: use the commit containing this file; its immutable SHA is recorded in the final Codex response.

## Outcome

- Default mode: `adaptive_gbif_fast_start`.
- Strict compatibility: `human_verified_strict` remains available, migratable, and covered by 86 passing strict tests.
- Full regression: 2,531 passed in 109.26 seconds.
- Acceptance: 65 passed, one user-authorized branch-policy deviation, zero unaccounted failures.
- Scientific release: blocked until the live steps and human review below are complete.

The specification named `codex/adaptive-gbif-reference-default`, but the user explicitly required all commits and pushes on `main`. Criterion 60 therefore remains visible as an authorized deviation; it is not silently reported as passed.

## Schema and policy contract

| Contract | Version |
|---|---|
| `admission_policy` | `reference-admission-policy-v1.0.0` |
| `admission_compiler` | `reference-admission-compiler-v1.0.0` |
| `support_manifest` | `reference-support-manifest-v3.0.0` |
| `legacy_support_manifest` | `reference-support-manifest-v2.0.0` |
| `support_migration` | `reference-support-manifest-migration-v1.0.0` |
| `readiness` | `reference-bank-readiness-v3.0.0` |
| `readiness_policy` | `reference-bank-readiness-policy-v1.0.0` |
| `bank_summary` | `reference-bank-summary-v2.0.0` |
| `model_input_identity` | `reference-model-input-identity-v2.0.0` |
| `embeddings` | `reference-embeddings-v3.0.0` |
| `embedding_checkpoint` | `reference-embeddings-checkpoint-v2.0.0` |
| `embedding_report` | `reference-embeddings-report-v1.0.0` |
| `pilot_report` | `adaptive-pilot-report-v1.0.0` |
| `release_verification` | `gbif-adaptive-release-verification-v1.0.0` |

Provisional scoring means raw BioCLIP similarity and ranking for screening only. It is not probability, confidence, calibration, human verification, or scientific-release authorization.

## Verification

| Gate | Result |
|---|---|
| Full suite | 2,531 passed; 0 failed; 109.26 s |
| Strict mode | 86 passed; 18.45 s |
| Adaptive mode | 65 passed; 7.68 s |
| CLI | 143 passed; 7.70 s |
| Pilot and fixtures | 66 passed; 8.46 s |
| Performance regression | Passed; six measured fixture dimensions |
| Ruff lint | Passed |
| Locked dependency audit | 0 known vulnerabilities |
| Secret review | 122 heuristic hashes/placeholders/fake test values classified; 0 private-key/common-live-token-prefix matches |
| Artifact parse | 86 JSON, 3 JSONL, 1 Parquet; 0 errors |
| Live source smoke | Not run—credentials and durable corpus unavailable |
| Ruff format | Unconfigured failing baseline: 308 files would change |
| mypy | Unconfigured failing baseline: 892 errors in 114/247 package files |

The format and type results are open toolchain debt, not hidden waivers. The live-source omission means no live network, corpus-quality, or species-performance claim is made.

The first final-boundary run found one governance test coupled to exact phrases
from the replaced monolithic `AGENTS.md` (2,530 passed, one failed in 110.35 s).
The test now accepts equivalent approved wording from either the monolithic or
root-plus-topic instruction layout while enforcing the same seven scientific
boundaries. The subsequent full run passed all 2,531 tests.

## Benchmark and pilot results

The fixture-backed Papilio path reached its first provisional score in **502.480375 ms** after **zero** reference reviews. It considered and provisionally admitted one GBIF fixture reference, reused one reference and one Flickr embedding, and selectively rescored one Flickr fixture record. The targeted-remediation fixture flags one species, leaves one unaffected, reviews and excludes one reference, and reruns only the affected species. These are mechanism tests, not live Papilio outcomes.

The current audit covers one species but has zero of 50 required human-reviewed Flickr labels. Consequently, zero species have sufficient samples, all four quality metrics are null, no live species is flagged, and no live reference remediation or selective rerun occurred. One fixture Flickr record was machine-scored; zero current records are human-reviewed or eligible for final export; **zero unreviewed records were exported**.

Historical prototype-only context records 13,496 scored Flickr records, 81 reused reference embeddings, one persistent model load, six model-cache hits, and 2.274524 records/s. It is not the current pilot and is not evidence of a production speedup. Strict time to first score, YOLOE reruns avoided, and common-unit full-rerun savings were not measured and remain unavailable.

## Live and human work still required

- Acquire and admit a current-policy GBIF support bank from durable media objects.
- Generate current-policy BioCLIP reference embeddings from those durable objects.
- Build current-policy provisional route-separated prototypes from support_train only.
- Score the current Flickr workload with current-policy provenance and resource instrumentation.
- Complete 50 source-bound representative Flickr reviews.
- Run the species audit, then targeted reference review and selective rerun only if legitimately flagged.

- Complete the 50-record representative Flickr queue; skip, can't-view, uncertain, and unresolved conflicts do not count.
- Human-verify every Flickr record admitted to a final occurrence export.
- Review only references targeted after sufficient human-labelled Flickr evidence; provider assertions remain provisional.

Provider-asserted GBIF support remains provisional. Skip, can’t-view, uncertain, and unresolved Flickr decisions never count as verified.

## Claims boundary

**Allowed:**
- Adaptive policy and strict compatibility are implemented and tested.
- The fixture path reaches provisional scoring in 502.480375 ms with zero prior reference reviews.
- Selective review, cache reuse, rebuild, rescore, and fail-closed export work in deterministic fixtures.

**Blocked:**
- Provider-asserted GBIF references cannot be called verified.
- No live Papilio accuracy, quality gain, production speedup, memory saving, or review saving is established.
- Raw similarity and provisional margin are not probability or confidence.
- No scientific release is authorized without Flickr labels and live audit evidence.

## Remaining limitations

- Live GBIF acquisition, durable-media admission, scoring, and provider smoke tests were not executed.
- Pilot evidence is fixture-backed; historical prototype counts are context only.
- Fifty representative Flickr records remain unreviewed, so quality metrics are unavailable.
- Ruff formatting and mypy are unconfigured and their failing baselines remain debt.
- YOLOE reruns avoided and common-unit full-rerun savings were not instrumented and remain null.
- Main is a user-authorized deviation from criterion 60.

## Recommended integration procedure

1. Do not merge automatically: the user-directed implementation is already on main.
2. Review both final and release-verification reports, the 66-criterion matrix, and GitHits ledger.
3. Complete or explicitly defer live/human work; never convert fixture results into live claims.
4. Require maintainer and scientific-review approval before tagging.
5. Fetch origin/main, verify the report commit, rerun locked release gates on the exact SHA, then tag without rewriting history.

No automatic merge is recommended or required: the user-directed commits already reside on `main`.

## Task commit and push ledger

Every preceding task commit below is an ancestor of `origin/main` at `477eaface3d1f5efa51255550f0ef8d6a7740f35`. The final report task resolves to the commit containing this file and is verified after push in the final Codex response.

| Task | Commit | Push |
|---|---|---|
| `gbif-fast-0.1` | `7ca83acb7d0ada7fd026bd4c7f343939c6b72293` | verified on `origin/main` |
| `gbif-fast-0.2` | `247b42f3206d48bb79e2dbf97c5a92e4f207ae71` | verified on `origin/main` |
| `gbif-fast-0.3` | `a39b05c8ca843ae368c29aaa6ab3ebe0e459111b` | verified on `origin/main` |
| `gbif-fast-0.phase` | `c666eef04ef7f292562ef51acf54f05b8e91c6ab` | verified on `origin/main` |
| `gbif-fast-1.1` | `da7fb1944011dd2ee5e7e7aec53f123a9c589fc0` | verified on `origin/main` |
| `gbif-fast-1.2` | `282696d8d094a1af8c2be63ea0afa6d972e73e47` | verified on `origin/main` |
| `gbif-fast-1.3` | `9170f05c6b79b9d3ef2c9b528c4d261f53369bc8` | verified on `origin/main` |
| `gbif-fast-1.4` | `c15016893462c93683185959ab1277e5393899b5` | verified on `origin/main` |
| `gbif-fast-1.phase` | `31aafd3b35ba28bd694a495c36a4d386c4e297db` | verified on `origin/main` |
| `gbif-fast-2.1` | `2bc8f146a4745230786a30ef0f23773c36f4c5c0` | verified on `origin/main` |
| `gbif-fast-2.2` | `3dbdb0ce118478e8a1eac16b1b50b0e76e39d5b4` | verified on `origin/main` |
| `gbif-fast-2.3` | `782c84c38725d025272acf5383569ed5f8108de1` | verified on `origin/main` |
| `gbif-fast-2.4` | `b9d790baeba3b3597bf1973d83a5650883f5ec57` | verified on `origin/main` |
| `gbif-fast-2.phase` | `26778b3e79f7ab6c79d4dc5d4606a31491db4b47` | verified on `origin/main` |
| `gbif-fast-3.1` | `e239b4c3694e466041f39445e7d276e63b251ed1` | verified on `origin/main` |
| `gbif-fast-3.2` | `053473526316890520bfe85a1ea00f1d92a23efc` | verified on `origin/main` |
| `gbif-fast-3.3` | `7b81a954943a2fca46a13dfbd0e06b1a711439c7` | verified on `origin/main` |
| `gbif-fast-3.4` | `e063d7e7dfa7745aa74f66cd5b6a3e65ce9fd3db` | verified on `origin/main` |
| `gbif-fast-3.phase` | `7f57fb71300925a822393bcf53366e6c7cc7f3ac` | verified on `origin/main` |
| `gbif-fast-4.1` | `fe882bd8f0b4749cd10f8cea28a6bbcc2f7dfbb3` | verified on `origin/main` |
| `gbif-fast-4.2` | `d5103b5842bbb25555dd6f91b68c1962cd40322a` | verified on `origin/main` |
| `gbif-fast-4.3` | `df49d671cb3728172c9fce2fc4de029d1adeb21d` | verified on `origin/main` |
| `gbif-fast-4.4` | `0a428d5ad3f5008e162f0a9a94da3dcbfede3797` | verified on `origin/main` |
| `gbif-fast-5.1` | `fda9390e8729ff7d633a9ea71f169c532329cf07` | verified on `origin/main` |
| `gbif-fast-4.phase` | `e345880a02bd21dc7569e74b7912b42ab13e5469` | verified on `origin/main` |
| `gbif-fast-5.2` | `98f2e9dcddf0fe5dbb08d3d6dff4f2df0ca9df59` | verified on `origin/main` |
| `gbif-fast-5.3` | `725e77539da279332d4c8fd256a99eee44de545e` | verified on `origin/main` |
| `gbif-fast-5.4` | `410e19a0903532123000d9ef7fd6c4290e04b444` | verified on `origin/main` |
| `gbif-fast-5.phase` | `3c7665df3a828b2ea925ee8b549bf843c569f540` | verified on `origin/main` |
| `gbif-fast-6.1` | `5f38036c1ba4016e8b15ae9e4543ec244c676069` | verified on `origin/main` |
| `gbif-fast-6.2` | `d40a600a461da1f15cd0cf63f72387650d63657c` | verified on `origin/main` |
| `gbif-fast-6.3` | `c2ff7cc722f8ebd8700db2bac86366c219198856` | verified on `origin/main` |
| `gbif-fast-6.4` | `2643bf3c1caffe2f68ba837d99064a0c99192c7c` | verified on `origin/main` |
| `gbif-fast-6.regression-sqlite` | `096380ad86324102dee67e3f73cc71ce21798036` | verified on `origin/main` |
| `gbif-fast-6.phase` | `3aaddece117d311db297b135be510eea0338e55a` | verified on `origin/main` |
| `gbif-fast-7.1` | `d71bceabf75748a25df39d0025e8da907f295f8c` | verified on `origin/main` |
| `gbif-fast-7.2` | `b434f4df9707a16469b7371a1eea51ae65ac74fb` | verified on `origin/main` |
| `gbif-fast-7.3` | `7111f664e1edc4d967d069390dd8ec40454288fe` | verified on `origin/main` |
| `gbif-fast-7.4` | `843aeb3dc604ee28c4d1d260447e85a1a451f3b2` | verified on `origin/main` |
| `gbif-fast-7.phase` | `f54db40a09cf03832d4cced6a4aed1d3fed1d3b1` | verified on `origin/main` |
| `gbif-fast-8.1` | `4a9a7fa53af26b1b6414536a93def82b6076f6ca` | verified on `origin/main` |
| `gbif-fast-8.2` | `534b2894b24f393367555e9a04ad73da31609bb9` | verified on `origin/main` |
| `gbif-fast-8.3` | `d2eead538cb21a2d6caeb8a138ebea6b22a7aea9` | verified on `origin/main` |
| `gbif-fast-8.4` | `7146807185e7d6406b9f0610d4f8d06623ece79a` | verified on `origin/main` |
| `gbif-fast-8.phase` | `d64e55369cc3030028b3ba730568042802d6e4f3` | verified on `origin/main` |
| `gbif-fast-9.1` | `622e97f80c181e098d78970bf56f534f36ec2ef2` | verified on `origin/main` |
| `gbif-fast-9.2` | `b9a798a3e8e38eacf5e11ceaccccd242a7e6c780` | verified on `origin/main` |
| `gbif-fast-9.3` | `0208c0fbc2e9e92374ac0cadc49643113d5d0c8b` | verified on `origin/main` |
| `gbif-fast-9.4` | `efd8d1ce3b5d3a81932e16fbdee246caa715dbf9` | verified on `origin/main` |
| `gbif-fast-9.5` | `0fd1bce06788e1adfef7b1d11bdcae2e7284ca7e` | verified on `origin/main` |
| `gbif-fast-9.phase` | `c628f324088998824205955a13304c38adf36472` | verified on `origin/main` |
| `gbif-fast-10.1` | `77c19cf478e6ff7efeb31029a7b9d729aa4c0a37` | verified on `origin/main` |
| `gbif-fast-10.2` | `66de5f5e6202be533bb704b39a131d5c72bd718d` | verified on `origin/main` |
| `gbif-fast-10.3` | `3b51b86715aec6750fd185421998ee660eb59a10` | verified on `origin/main` |
| `gbif-fast-10.4` | `83f25328c6b44b33d9c97ece1658fb622b2c73ea` | verified on `origin/main` |
| `gbif-fast-10.phase` | `787aff8ea7a931d6d9b845b6b4d2cf1aa620debb` | verified on `origin/main` |
| `gbif-fast-11.1` | `883780a1841de80622bcdaf5dfacb92c29c28f91` | verified on `origin/main` |
| `gbif-fast-11.2` | `23eb833c3fad0c2df74a5e6711fd7823bbfe2a12` | verified on `origin/main` |
| `gbif-fast-11.3` | `2e807546c0d4cebe50c18f1a2b8df63aa9877d3d` | verified on `origin/main` |
| `gbif-fast-11.4` | `ddc6e3ea036a953f1a9f7875feb54543204b532c` | verified on `origin/main` |
| `gbif-fast-11.phase-repair` | `3962db990944bb03e2627472ba5bcb6a8ec7224d` | verified on `origin/main` |
| `gbif-fast-11.phase` | `0ee83f50c780ecb0615548eadb5f50384395ad72` | verified on `origin/main` |
| `gbif-fast-12.1` | `3fe2b88a6d30ac922c0e02331f1b39e7f71d7225` | verified on `origin/main` |
| `gbif-fast-12.2` | `9e2e2c2ecf6f1fcf737a04bf38b76f3d9d52c351` | verified on `origin/main` |
| `gbif-fast-12.3` | `5b3d384339b1325cb04c34b6c5f5cd339b4cd86f` | verified on `origin/main` |
| `gbif-fast-12.4` | `32a043684f863bed5d2eeab615b63b27872d2774` | verified on `origin/main` |
| `gbif-fast-12.5` | `67d2137c06c24f8e87ce3d081dc0dd215d49db45` | verified on `origin/main` |
| `gbif-fast-12.phase-repair` | `d53f57ea518a8fac13ce6f4216f784a8f38078ca` | verified on `origin/main` |
| `gbif-fast-12.phase` | `5f5ba54010f758843b8a972150da136665a9d7b4` | verified on `origin/main` |
| `gbif-fast-13.1` | `213d2fc7e0f07bf48ef0d35b7057e9707a321346` | verified on `origin/main` |
| `gbif-fast-13.2` | `6c0a25ae19f22a2b31c114774e3a03a02df5c878` | verified on `origin/main` |
| `gbif-fast-13.3` | `c47a004bc2bcc11759df51fc8851d3b1b74788ee` | verified on `origin/main` |
| `gbif-fast-13.4` | `477eaface3d1f5efa51255550f0ef8d6a7740f35` | verified on `origin/main` |

## Acceptance audit

The machine-readable companion contains all 66 criteria with evidence groups. Criteria 1–59 and 61–66 pass. Criterion 60 is the sole authorized deviation because the user required `main`; no external code was copied, every task has a GitHits record, and recorded GitHits outages remain explicit.

GitHits final-report review: solution `7a90d56d-11ee-42d1-a8d3-3d6a2c2c823a`, distilled from MIT and Apache-2.0 sources. BioMiner adopted only the evidence-ledger structure and copied no external code or prose.
