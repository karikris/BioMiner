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
- Sort canonical rows and normalize identity-bearing lists to their semantic
  order before serialisation. Ordered rank paths are never lexically sorted.
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

### Post-implementation verification — 2026-07-11

The post-implementation queries targeted the operations actually present in
`classification_v3.py`; all example searches used strict licence mode.

Directed-graph call:

```text
get_example(
  language="Python",
  license_mode="strict",
  format="json",
  query="Python deterministic directed graph validation implementation using depth-first search with active recursion path for cycle detection, sorted node and child traversal, and enforcing that every enabled child has at most one enabled parent"
)
```

Solution ID: `b4cedb7f-90a3-4a70-a5f9-889932e7b78a`.
Immutable sources:

- `abhisek2004/45-Days-Python-Development-Challenge@e8b37e63eca01feba74c10d6a2e006341178e113`
  (MIT);
- `cy-suite/InvokeAI@6b18f270dda57e74ff3d9640d9caf131d20f0c3d`
  (Apache-2.0).

Adopted: separate visited/active DFS state, sorted roots and children, parent
identity deduplication, and enabled-endpoint filtering for the single-parent
invariant. Reviewed rank-skip edges remain real graph edges. Rejected:
dictionary/set iteration as output order, excluding skip edges, and treating
duplicate copies of one edge as distinct parents.

Canonical-fingerprint call:

```text
get_example(
  language="Python",
  license_mode="strict",
  format="json",
  query="Python deterministic content fingerprint for a list of record dictionaries: normalize nested dictionaries and lists, sort records independently of input row order, serialize canonical JSON using sorted keys and compact separators, then SHA-256 digest"
)
```

Solution ID: `e75c1d38-3897-418b-800a-5fdeefc6c825`.
Immutable sources:

- `ai-blaise/nmoe@f8a28d78321e1aae8f8e8bb0a799351624b2b92e`
  (Apache-2.0);
- `Noumena-Network/nmoe@970a146433f9c649d09ddab36f675974f53dd905`
  (Apache-2.0).

Adopted: recursive dictionary canonicalisation, explicit row sorting,
`sort_keys=True`, compact JSON separators, UTF-8 and SHA-256. Rejected: Python
`hash()`, Polars `hash_rows()`, MD5, and sorting semantically ordered nested
lists. The implemented tests prove row-order invariance; changes to ordered
`rank_path`, `rank_path_node_ids`, or `skipped_ranks` remain identity changes.

Pinned Polars source verification targeted repository
`pola-rs/polars` at tag `py-1.41.2`, immutable commit
`599a503a0997188a74750926a5cdaa47585cf8aa`. GitHits reported exact-current
resolution, and `code_read(LICENSE, 1:30)` confirmed MIT. Exact calls and
findings:

- `code_grep("class List", path_prefix="py-polars/src/polars/datatypes")`
  followed by `code_read(classes.py, 1050:1115)` confirmed that `List` is the
  variable-length nested type whose constructor accepts an inner datatype.
  This verifies `pl.List(pl.String)`. Indexed content hash:
  `ecd0fa8bae9f3cca968dbafa40203c265cbfe1630e5d7d2025ff33f1b0552489`.
- `code_grep("strict: bool = True", path_prefix="py-polars/src/polars/dataframe")`
  followed by `code_read(frame.py, 365:445)` confirmed explicit `schema` and
  `strict` constructor parameters and their forwarding for sequence input.
  Indexed content hash:
  `c6b96fe46fff30450f01c3f6c725aefab32ab0e4416024e43efc2b22543cd9aa`.
- `code_grep("def group_by(", path_prefix="py-polars/src/polars/dataframe")`,
  `code_read(frame.py, 7120:7200)`, `code_grep("def len(",
  path_prefix="py-polars/src/polars/dataframe/group_by.py")`, and
  `code_read(group_by.py, 572:630)` confirmed list-valued grouping keys,
  `GroupBy.len()` naming its count column `len`, and the lack of output-order
  guarantees when `maintain_order=False`. GroupBy content hash:
  `7518b84acca31ae537913d41e9a99f036d92195c42da5d2de0fc1d00b5ee4d4f`.

Implementation comparison:

- `_typed_frame` supplies exact schemas and type-correct list defaults. Its
  deliberate `strict=False` coercion is bounded by normalization and strict
  cross-table QA rather than trusted as source validation.
- `_append_duplicate_findings` uses the verified
  `group_by(...).len().filter(pl.col("len") > 1)` API and explicitly sorts
  findings afterward; it does not pay for or depend on `maintain_order=True`.
- `_first_cycle` sorts roots and children, while `_validate_edges` uses sets
  only for membership/deduplication and sorts them before persistence.
- `_canonical_rows` preserves within-row semantic list order but sorts records
  independently of source or DataFrame row order.

Repository verification:

- `uv run ruff check src/biominer/registry tests/test_registry_classification_v3.py`:
  passed;
- `.venv/bin/pytest -q tests/test_registry_classification_v3.py`:
  `18 passed`;
- `.venv/bin/pytest -q`: `884 passed`.

Post-implementation solution IDs:

`b4cedb7f-90a3-4a70-a5f9-889932e7b78a`,
`e75c1d38-3897-418b-800a-5fdeefc6c825`.

## Phase 2 — Path-aware taxonomy store

### Pre-implementation verification — 2026-07-11

Research questions:

- How should enabled nodes be extracted from a denormalized active-path table?
- When is a semi-join preferable to `is_in`?
- Which uniqueness and ordering guarantees are explicit in pinned Polars?
- Does a moderate, already-loaded taxonomy table benefit from LazyFrame APIs?

All four example queries used `language="Python"`, `license_mode="strict"`
and `format="json"`.

1. Path list filtering, exploding, uniqueness and sorting.
   Solution ID: `1b65243e-51a3-4304-9821-eab4f2570fe0`.
   Immutable source:
   `Neural-Dragon-AI/Cynde@332b79f1de1d7671e917f5034a63cfbfc3b74e23`
   (Apache-2.0).
2. Deterministic semi-join filtering.
   Solution ID: `511fc304-02b1-49b4-872f-1b5882d08a10`.
   Immutable sources included
   `rapidsai/cudf@3aa656886f0d450cac82a69c095fc0d6b9b97ca2`
   (Apache-2.0),
   `graphistry/pygraphistry@8a7668e43aff3c7ee1773d809b39bd1a1c218dbf`
   (BSD-3-Clause), and
   `narwhals-dev/narwhals@272197529389f123972c8610e15ebc46014c7f4c`
   (MIT).
3. Lazy collection and common-subplan elimination.
   Solution ID: `95dff618-be06-4f89-8098-4a80075eef2d`.
   Immutable sources included
   `pola-rs/polars@57f20cae8a79db98dac5ee5cea1ebfacbf74dae5`
   (MIT) and
   `henryodibi11/Odibi@fc613c2d8dcdf3920e435c82403d90f7385ce00f`
   (Apache-2.0).
4. Extracting distinct identifiers across several columns with explicit final
   sorting.
   Solution ID: `95fc44d5-7d13-42f2-b2b0-6e836ecb9cbd`.
   Immutable sources included
   `narwhals-dev/narwhals@272197529389f123972c8610e15ebc46014c7f4c`
   and `goldenmatch/goldenmatch@38728c14c5e8fa994b5822c93bc3b65ad29a5540`
   (both MIT).

Pinned dependency verification again targeted `pola-rs/polars` tag
`py-1.41.2`, immutable commit
`599a503a0997188a74750926a5cdaa47585cf8aa`; `code_read(LICENSE)` confirmed
MIT. Exact source reads established:

- `DataFrame.join` in `py-polars/src/polars/dataframe/frame.py:8206` accepts
  `semi` and `anti`; its documentation at line 8280 states that output order
  is unspecified unless requested and that leaving it unspecified permits
  more optimisation.
- `DataFrame.unique` at line 11173 documents that `keep="any"` does not select
  a predictable retained row, while `maintain_order=True` is more expensive
  and blocks streaming.
- `collect_all` in `py-polars/src/polars/functions/lazy.py:2058` combines query
  graphs and applies common-subplan elimination.
- `Expr.list.contains` and `Expr.list.explode` are implemented in
  `py-polars/src/polars/expr/list.py:779` and `:1115`.
- Exact eager and lazy semi/anti join tests exist in
  `py-polars/tests/unit/operations/test_join.py:29`.

Patterns adopted:

- Load projected v3 Parquet frames once and use eager queries for this
  moderate, repeatedly queried store.
- Extract nonblank stable node IDs from direct rank columns, deduplicate by
  `node_id`, then semi-join against enabled canonical nodes.
- Use `is_in` for one small scalar ID set and semi-joins when IDs are already
  a frame or keys become composite.
- Explicitly sort every public result; no query relies on join, group, input,
  or set iteration order.
- Preserve distinct IDs even when scientific names collide.

Patterns rejected:

- The first generated example claimed that filtering an active leaf before
  exploding proves every ancestor is active. BioMiner rejects that inference:
  extracted IDs must still be joined to enabled canonical nodes.
- `any_horizontal(list.contains(...))` scales the expression graph with the
  beam size; direct rank-column filtering is simpler for top-k node IDs.
- `unique(keep="any")` is not used when duplicates could differ.
- Lazy execution is not assumed faster. `collect_all` is reserved for future
  shared scans or branching plans demonstrated by profiling; it adds needless
  collection boundaries after the store has already loaded moderate frames.
- PR- or issue-only links without immutable source commits were not accepted
  as durable implementation evidence.

Pre-implementation solution IDs:

`1b65243e-51a3-4304-9821-eab4f2570fe0`,
`511fc304-02b1-49b4-872f-1b5882d08a10`,
`95dff618-be06-4f89-8098-4a80075eef2d`,
`95fc44d5-7d13-42f2-b2b0-6e836ecb9cbd`.

### Post-implementation verification — 2026-07-11

The strict post-implementation searches mirrored the store operations exactly:

1. `get_example`: “Python Polars DataFrame semi join candidate rows against a
   one-column key DataFrame, preserving only matching left rows”.
   Solution ID: `78ded640-7add-4677-8a43-d652116d3eed`.
   Durable source:
   `graphistry/pygraphistry@8a7668e43aff3c7ee1773d809b39bd1a1c218dbf`
   (BSD-3-Clause); mutable issue references were supplemental only.
2. `get_example`: “Python Polars filter scalar column membership with is_in
   and filter list column with list.contains”.
   Solution ID: `ca6b81e3-0f4d-41d3-8af8-1ee3fd6c3b3b`.
   Durable sources included
   `pola-rs/polars@57f20cae8a79db98dac5ee5cea1ebfacbf74dae5`
   and `narwhals-dev/narwhals@272197529389f123972c8610e15ebc46014c7f4c`
   (MIT), plus the immutable Graphistry source above.
3. `get_example`: “Python Polars deterministic deduplicate rows using sort
   then unique subset keep first maintain_order”.
   Solution ID: `fc033682-87c8-47c9-8ad1-4f320d14f4a3`.
   Durable sources:
   `yedivanseven/swak@96394dae9a2161bbd4b1b8c35aab6353f1535119`
   and `phurwicz/hover@dccdaaeb616b87d56c0b7bc83c72b781c402e6a8`
   (MIT).
4. `get_example`: “Python Polars lazily scan Parquet, project selected
   columns, then collect with the streaming engine”.
   Solution ID: `561375ba-a8e2-4200-a275-c494f366acb6`.
   Durable sources included
   `rapidsai/cudf@3aa656886f0d450cac82a69c095fc0d6b9b97ca2`
   and `marimo-team/marimo@1f1bb633df7e635899ffb88479576944dd15b543`
   (Apache-2.0), plus
   `ancoleman/ai-design-components@76551b7b19ebc667764ec75da14990d0aef8b6e5`
   (MIT).

Pinned Polars verification used the exact Phase 1 target again:
`pola-rs/polars@599a503a0997188a74750926a5cdaa47585cf8aa`
(`py-1.41.2`, MIT). `code_grep` and `code_read` confirmed:

- `DataFrame.join(..., how="semi")` at `frame.py:8207-8255`;
- `DataFrame.unique(subset=..., keep="first", maintain_order=...)` at
  `frame.py:11173-11199`;
- `Expr.is_in(Collection)` at `expr.py:6405-6447`;
- `Expr.list.contains` at `list.py:779-811`;
- Parquet scan projection pushdown at `io/parquet/functions.py:467-502`;
- `LazyFrame.collect(engine="streaming")` at
  `lazyframe/frame.py:2406-2423` and `:2482-2500`;
- same-schema vertical concatenation in `functions/eager.py:32-115`.

Implementation comparison:

- Candidate extraction uses a one-column semi-join and an explicit final
  `(scientific_name, node_id)` sort.
- Small selected node beams use scalar `is_in`; reviewed skip detection uses
  the typed `List(String).list.contains` operation.
- Path unions use ordinary vertical `pl.concat` because exact schemas are a
  store invariant. Relaxed or diagonal concatenation is rejected.
- Deduplication sorts first, then uses `hierarchy_hash`, `keep="first"` and
  `maintain_order=True`, followed by another explicit semantic sort.
- Artifact loading uses projected `scan_parquet` and the pinned
  `collect(engine="streaming")` spelling. A generated example’s deprecated
  `collect(streaming=True)` spelling was rejected.
- Join order, `unique(keep="any")`, and `maintain_order` without deterministic
  pre-sorting were rejected as identity/order mechanisms.

The post-review comparison found that Polars drops null filter predicates, so
`filter(~pl.col("enabled"))` did not reject `enabled=null`. The final store now
requires zero nulls and `Series.all() is true`. The same review also led to:

- exact physical Parquet schema validation before collection;
- binding active paths to the store’s validated hierarchy-hash set;
- cached access to already validated manifest fingerprints rather than full
  graph recanonicalisation on every property access;
- a read-only top-level manifest mapping.

Repository verification:

- `uv run ruff check src/biominer/bioclip/path_taxonomy_store.py tests/test_path_taxonomy_store.py tests/test_registry_classification_v3.py`:
  passed;
- `.venv/bin/pytest -q tests/test_path_taxonomy_store.py tests/test_registry_classification_v3.py tests/test_five_rank_taxonomy_store.py`:
  `44 passed`;
- `.venv/bin/pytest -q`: `907 passed`.

Post-implementation solution IDs:

`78ded640-7add-4677-8a43-d652116d3eed`,
`ca6b81e3-0f4d-41d3-8af8-1ee3fd6c3b3b`,
`fc033682-87c8-47c9-8ad1-4f320d14f4a3`,
`561375ba-a8e2-4200-a275-c494f366acb6`.
