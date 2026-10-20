# Arrow 25 security backport

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

The builder accepts only the pinned Arrow 25 archive, this exact patch, and
the pinned original and resulting hashes of both affected files. It does
not accept a partial patch on resume. The custom wheel suffix binds the
preserved recipe and security patch through immutable `build-inputs.json`.
Source preparation alone does not qualify a compiled wheel or runtime.
