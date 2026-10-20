# Arrow 25 backports

`arrow25-a769c291.patch` is the unmodified upstream patch from
[Apache Arrow commit a769c291e01093b73d03a075179cf7a09bf92ad8](https://github.com/apache/arrow/commit/a769c291e01093b73d03a075179cf7a09bf92ad8),
merged through [PR 50843](https://github.com/apache/arrow/pull/50843).
Its SHA-256 is
`d51722f4ddb3934a0c846b691a60a730cb77d5a814d7077cab1c5d145c54aaa1`.

The patch preserves upstream authorship and the Apache-2.0 source license.
It changes the chunked boolean allocation paths for `replace_with_mask`,
`fill_null_forward`, and `fill_null_backward`, and adds the three upstream
C++ regression tests. Keep the original Arrow LICENSE and NOTICE files with
the source and distributed artifacts.

`arrow25-0efbdec7.patch` is the unmodified upstream patch from
[Apache Arrow commit 0efbdec7a10da1a4043fc1210c61e73c88bc9b4f](https://github.com/apache/arrow/commit/0efbdec7a10da1a4043fc1210c61e73c88bc9b4f),
merged through [PR 51240](https://github.com/apache/arrow/pull/51240) for
GH-51223. Its SHA-256 is
`6c64e115aa53621ae7e1992417b26ab7c6c51fb265d3e64216580f7dcae87d50`.
It includes a sliced Boolean array's offset when copying packed values and
adds upstream regression tests for the same three compute operations.
It also preserves upstream authorship and the Apache-2.0 source license.

The builder applies the allocation patch first and the slice-offset patch
second to the pinned original Arrow 25 archive, with zero fuzz and no offsets.
The receipt binds each stage's exact before/after hashes, including the shared
test file's intermediate hash. A separate `final_sources` inventory binds the
three unique resulting files; only these final bytes enter retained source
inputs. Missing, reordered, extra or changed patch records and partial source
states fail on resume. The custom wheel suffix binds the preserved recipe,
both patches and the final source inventory through immutable `build-inputs.json`.
Source preparation alone does not qualify a compiled wheel or runtime.
