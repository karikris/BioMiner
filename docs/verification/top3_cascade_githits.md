# Global top-3 cascade: GitHits verification

This log records external implementation evidence used while replacing the
five-rank cumulative path beam. GitHits evidence supplements BioMiner's own
tests and reviewed taxonomy QA; it does not define taxonomy or production
classification policy.

## Phase 1 — Immutable six-rank taxonomy artifacts

### Pre-implementation verification — 2026-07-11

Research questions:

- How do maintained Python projects validate directed graphs, cycles and
  single-parent constraints?
- Which parts of optional-rank skip assertions belong in graph identity?
- How should graph content be canonicalised before hashing?
- What list/struct and deterministic-unique behaviour is supported by the
  Polars version pinned by BioMiner?

Repository dependency evidence:

- `uv.lock` pins `polars==1.41.2` and `polars-runtime-32==1.41.2`.
- `uv.lock` pins `pyarrow==24.0.0`.
- The active environment reports Python 3.14 and the project requires
  Python 3.14 or newer.

GitHits calls and results:

1. `get_example`, strict licence mode, Python: versioned taxonomy graph,
   reviewed skip edges, cycle detection, single enabled parent and stable
   fingerprinting.
   Solution ID: `78a5fd65-7fc3-4457-b954-1d1a8e242fa2`.
   Sources:
   `PrefOptimize/PrefXplain@d05c1469d3d27bfefeaec1316d82fada6810ecfe`
   (Apache-2.0) and
   `sipyourdrink-ltd/bernstein@4d5a257f0d57bb1a329e3e107000d26c2fd90801`
   (Apache-2.0).
2. `get_example`, strict licence mode, Python: deterministic DAG validation,
   canonical adjacency and stable content hashes.
   Solution ID: `a04caaf5-9c66-4bcf-9e7b-8d08dbe3a062`.
   Sources included immutable MIT and Apache-2.0 commits, notably
   `mattmre/CHELATEDAI@34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`,
   `Mg30/flowrun@d67531e5ac6a9f0e4c8026dafbfd1cf600e155a7`, and
   `HKUSTDial/DeepEye@54709e664c8ce64eea0d6603a575ab302cb06a61`.
3. `pkg_info` for PyPI `polars` confirmed the upstream repository and that
   1.41.2 is a published release. This latest-package query was not used as
   evidence for pinned API behaviour.
4. `search` against PyPI `polars@1.41.2` for list/struct schema and unique
   operations returned `UNRESOLVABLE` because the package version could not
   be mapped to a repository ref. The required single retry targeted upstream
   tag `py-1.41.2` and resolved it to immutable commit
   `599a503a0997188a74750926a5cdaa47585cf8aa`.
5. `get_example`, strict licence mode, Python: explicit Polars list/struct
   schemas, deterministic sorting and canonical serialisation.
   Solution ID: `b6ea6462-b3ec-441b-8a56-39bc29c59fb8`.
   Sources were MIT, BSD-3-Clause and Apache-2.0, including
   `mlflow/mlflow@20858c9426a9644dfcf9545a92cbb534376779eb` and
   `apache/hamilton@f1dde11af305b3df3f0304a8ac14c11bfb04a904`.
6. `code_grep`, `code_files` and `code_read` inspected Polars tag
   `py-1.41.2` at commit `599a503a0997188a74750926a5cdaa47585cf8aa`.
   Upstream's `LICENSE` is MIT. Exact source evidence confirmed:
   `pl.List` is an explicit variable-length nested type;
   struct rows are supported by `DataFrame.unique()`; and order-preserving
   unique behaviour is an explicit option rather than an implicit guarantee.

Constraints adopted:

- Use explicit `pl.List(pl.String)` fields and `[]` defaults. Never let a
  generic string default populate a nested column.
- Sort canonical rows and every identity-bearing list before serialisation.
- Use canonical JSON with sorted keys and SHA-256 for hierarchy and artifact
  fingerprints.
- Validate node existence, self-loops, cycles and enabled single-parent
  constraints independently of path materialisation.
- Treat a reviewed rank skip as an explicit, provenance-bearing graph edge.
- Preserve source-scoped stable node IDs; display names and release dates are
  not node identity.

Patterns explicitly rejected:

- The retrieved demonstration excludes skip edges from cycle detection.
  BioMiner rejects that pattern: a reviewed `TRIBE -> GENUS` skip is still a
  parent relationship and participates in cycle and single-parent QA.
- Topological ordering is not used as a substitute for validating the exact
  six-rank transition contract.
- `hash_rows`, Python object hashes and MD5 are not durable registry identity.
- Candidate/source input order is not trusted, even when an API offers
  `maintain_order=True`; BioMiner sorts typed outputs explicitly.
- Nested structs are not introduced into Phase 1 merely because Polars
  supports them. Typed list columns plus wide rank columns are simpler and
  sufficient for immutable v3 artifacts.

Effect on BioMiner's design:

- `classification_v3.py` will be a new module; v2 semantics remain unchanged.
- The v3 edge schema will distinguish `asserted_parent` from
  `reviewed_rank_skip` and make skipped ranks and the skip reason explicit.
- Missing `SUBTRIBE` is represented only by a reviewed skip assertion, never
  by a fabricated node or placeholder.
- Path hashes will include ordered source-scoped node IDs, edge types, skipped
  ranks and the accepted GBIF species key.
- QA will inspect ambiguous parents before constructing paths, avoiding v2's
  lossy child-to-single-parent dictionary behaviour.

Pre-implementation solution IDs:

`78a5fd65-7fc3-4457-b954-1d1a8e242fa2`,
`a04caaf5-9c66-4bcf-9e7b-8d08dbe3a062`,
`b6ea6462-b3ec-441b-8a56-39bc29c59fb8`.

