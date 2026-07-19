# Prompt-variant provenance default removal

Date: 2026-07-19

`PromptVariant` no longer invents `legacy-unversioned`, `legacy` template, or
`legacy` evidence identities when callers omit provenance. Every caller must
now provide:

- `prompt_version`;
- `template_id`; and
- `evidence_kind`.

The canonical taxonomic prompt builder already supplied these fields and its
output is unchanged. This cutover affects direct constructors and tests only;
missing provenance now fails at construction instead of entering fingerprints,
pooling evidence, or reports under an ambiguous compatibility label.

GitHits was not called under the user's explicit directive. Provenance records
`githits_status: skipped_user_directive` and `solution_id: null`.
