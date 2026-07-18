# Task 10.1 completion — leakage-safe reviewed Flickr splits

Status: completed and pushed to `origin/main` through
`12aaaae3249dafa9991581b76c7849c82497fdd6`.

## Delivered

- Decisive source-bound reviewed rows are unioned into transitive independence
  components across Flickr photo, owner, duplicate, observation/burst, real
  geographic cluster and source mirror identities. Explicit no-geo rows never
  share a synthetic geographic identity.
- Whole components are frozen into calibration, validation or final test.
  Supported and error component coverage is reserved for all three partitions;
  a sparse outcome fails closed. Policy, seed, weights, source register and
  exact assignments are fingerprinted and fully recomputed during validation.
- The 18-component fixture freezes to 7/5/6 items against 40/30/30 targets of
  7.2/5.4/5.4. All partitions contain both outcomes and zero components cross.

## Transparent correction

After the first implementation push, an additional fixture audit found the
allocator produced 2/8/8 items despite correct leakage and outcome coverage.
The cause was an absolute-residual balance score that did not compare relative
target fill. Commit `12aaaae` corrects the score, adds a target-tolerance test,
and preserves the original `4da369e` commit in history.

## Gate

- Leakage suite: 39 passed in 0.91 seconds.
- Full regression: 2,985 passed in 104.20 seconds.
- Changed-file Ruff: passed.
- Remote `origin/main` resolved to `12aaaae…` after the correction push.

## Claim boundary

These deterministic fixtures prove the software grouping and frozen-assignment
contracts, not the sufficiency of live reviewed evidence, absence of unknown
identity leakage, calibrator validity, threshold support, occurrence release or
publication.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
