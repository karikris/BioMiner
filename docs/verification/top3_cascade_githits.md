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

## Phase 3 — Global rank-local top-three cascade

### Pre-implementation verification — 2026-07-11

BioMiner’s required beam is a node beam: at each rank, construct the union of
eligible nodes under the currently active paths, score those nodes using only
the current rank, sort once, and retain at most three globally. It is not a
complete-path beam, top-three per parent, a diversity quota, or cumulative
path search.

Primary strict call:

```text
get_example(
  query="beam search implementation global top k candidate pruning deterministic tie breaking tests",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `ff07eee2-f883-4dce-a4cd-37c584518675`.
GitHits source navigation resolved
`mberlanda/quantik-core-py@refs/pull/15/head` to immutable commit
`bd468bfce026c02c7755d70f4a6c36d56256f84c`; `code_read(LICENSE, 1:30)`
confirmed MIT. Focused reads found:

- `_score_and_prune` in `src/quantik_core/beam_search.py:500-570` scores the
  combined candidate frontier, sorts once, and slices `[:beam_width]`;
- mutation-sensitive ranking fixtures in `tests/test_beam_search.py` around
  line 303;
- same-seed determinism tests around line 188.

A supplemental strict query for a globally sorted candidate list with
deterministic secondary keys returned solution ID
`01e56d9b-226c-474c-8d7e-85891e70c457`. Its durable source was
`microsoft/sammo@2137815ba9a6fb8c86e5c18cc47ecda8ad8f5e98` (MIT); mutable
issue/PR links in the generated reference set were not used as durable
evidence. A broader taxonomy-specific query returned no result above the
quality threshold, and three other broad calls were terminated after repeated
waits; no IDs or claims were invented for them.

Patterns adopted:

- restrict the next candidate pool to descendants represented by currently
  active paths;
- deduplicate candidates by stable taxonomy `node_id` before scoring;
- score the entire current-rank union once;
- sort by raw similarity descending, scientific name ascending, then stable
  node ID ascending;
- retain exactly the first three actual nodes globally;
- keep earlier-rank scores only as audit evidence;
- use fixtures that make an incorrect algorithm produce a different survivor,
  rather than relying on ordinary happy paths.

Patterns rejected:

- the source’s insertion-index tie-break: BioMiner input/traversal order is not
  identity, so scientific name and source-scoped node ID are required;
- game-tree mover sign changes, rollout evaluation, terminal-state rules, and
  multiplicity semantics;
- top-k independently under every parent;
- reserving one branch per parent or otherwise enforcing taxonomic diversity;
- multiplying, summing, averaging, or otherwise combining parent scores with
  current-rank scores;
- injecting a target species or requested branch into open classification.

Decisive regression designs derived from the evidence:

- Global versus per-parent: A1=.99, A2=.98, A3=.97, B1=.96, C1=.95 must retain
  A1/A2/A3 even though all share family A.
- Current versus cumulative: parent A=.10 with A1=.99/A2=.98/A3=.97 must beat
  parent B=.90 with B1=.80/B2=.79/B3=.78 at the child rank.
- Tie: equal scores under permuted input must always retain the same lexical
  scientific-name/node-ID top three.
- Restriction: a score-1.00 child below a pruned parent must never enter the
  next candidate pool.

Pre-implementation solution IDs:

`ff07eee2-f883-4dce-a4cd-37c584518675`,
`01e56d9b-226c-474c-8d7e-85891e70c457`.

### Post-implementation verification — 2026-07-11

Strict post call 1:

```text
get_example(
  query="Python implementation score one combined candidate frontier, globally sort by score, retain top k once, deterministic tie breaking with stable secondary key, with regression tests",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `0eaddf09-f3db-493a-8533-ea91a4138e5b`.
Generated references were `vllm-project/vllm` PR 47348 (Apache-2.0) and
`raikonenfnu/tokenspeed` issue 24 (MIT).

Strict post call 2:

```text
get_example(
  query="Python deterministic top k records sorted by descending numeric score then ascending name and stable identifier, independent of input order, with tests",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `e1eefff4-fa7c-405a-a9f2-fadd7208e6fe`.
Its generated reference was the same TokenSpeed issue. Both solution IDs are
useful supplemental confirmation, but mutable PR/issue locations are not
durable source provenance.

The post-verification therefore re-read the immutable Phase 3 source:
`mberlanda/quantik-core-py@bd468bfce026c02c7755d70f4a6c36d56256f84c`
(`exact_current`, MIT). Exact GitHits reads were:

```text
code_read(LICENSE, 1:30)
code_read(src/quantik_core/beam_search.py, 500:575)
code_read(tests/test_beam_search.py, 180:215)
code_read(tests/test_beam_search.py, 295:335)
```

The immutable source again confirmed a combined frontier, one global sort,
and one top-width slice, supported by determinism and mutation-sensitive
ranking tests.

Implementation audit:

- `score_rank_candidates` deduplicates by stable `node_id`, requests one label
  vector, aggregates only the current node’s prompt similarities, and sorts
  once by `(-raw_similarity, scientific_name, node_id)`.
- `classify_path_cascade` takes one `scores[:3]` slice for every intermediate
  rank and filters active paths once by those global IDs.
- No parent score is read during selection. Source search found no
  `cumulative`, `stage_scores`, `_PathState`, `child_candidates`, or per-parent
  pruning machinery in the new classifier.
- Species candidates are the deduplicated union under genus top three, sorted
  by species raw similarity alone, and limited to 20 only after all eligible
  species are scored.
- Reviewed skip paths are carried from the already active parent set without
  becoming candidates or consuming beam slots.

The code review found one mixed-optional hardening gap: when some SUBTRIBE
nodes existed, a malformed path with neither an asserted node nor reviewed
skip evidence could previously disappear during filtering. The final code now
requires
`asserted_path_hashes union reviewed_skip_hashes == active_path_hashes` before
scoring any optional rank. It also verifies that each retained species mapping
has matching prefixed accepted and bare GBIF keys.

Decisive tests prove:

- the three retained subfamilies can all belong to one family;
- current child score defeats a higher earlier-rank branch score;
- a pruned family and its score-1.00 descendants cannot re-enter;
- repeated leaf paths score each ancestor node once;
- genus retains at most three actual nodes;
- species outside genus top three are never sent to the scorer;
- mixed and fully skipped SUBTRIBE behavior does not consume beam positions;
- species top 20 is ordered by species score only;
- mapping cardinality, accepted status, and key consistency fail closed;
- result models are frozen and contain no complete-path selection score.

Repository verification:

- `uv run ruff check src/biominer/bioclip/path_cascade_classifier.py src/biominer/bioclip/path_taxonomy_store.py tests/test_path_cascade_classifier.py tests/test_path_taxonomy_store.py tests/test_registry_classification_v3.py`:
  passed;
- `.venv/bin/pytest -q tests/test_path_cascade_classifier.py tests/test_path_taxonomy_store.py tests/test_registry_classification_v3.py`:
  `57 passed`;
- `.venv/bin/pytest -q`: `923 passed`.

Post-implementation solution IDs:

`0eaddf09-f3db-493a-8533-ea91a4138e5b`,
`e1eefff4-fa7c-405a-a9f2-fadd7208e6fe`.

## Phase 4 — Raw OpenCLIP similarities and distinct species reranking

### Pre-implementation verification — 2026-07-11

The application metadata in `src/biominer/cli.py` identifies
`open_clip_torch==3.3.0` and BioCLIP revision
`191d741545e4c741cdef4b22c6eb69c945c1e592`; the root `uv.lock` deliberately
does not contain the sidecar-only vision dependencies. The expected dedicated
runtime is absent on this host. Moreover, the current setup script installs
OpenCLIP and PyTorch without exact pins, so 3.3.0 is the source target for this
API review, not proof of the absent runtime's installed version. GitHits
`pkg_info` resolved PyPI `open-clip-torch` 3.3.0 to
`mlfoundations/open_clip`, MIT. A package-scoped `search` resolved that release
to immutable source commit `30573618fc375b12f094ef64cb3a1391cf611c45`.

Primary strict call:

```text
get_example(
  query="OpenCLIP Python encode one image batch and one text prompt batch, L2 normalize both embeddings, compute raw cosine similarities or pre-softmax logits, keep tensors device-safe for Apple MPS and transfer scalar results to CPU",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `5028e074-f077-484a-b607-cae57531390b`. Its generated
references were MIT-licensed CLIP ecosystem issue examples. They were useful
as a search lead, but the implementation decision was verified against the
immutable OpenCLIP package source rather than mutable issue pages.

Three additional strict calls supplied independent implementation evidence:

- `7e71f8b6-3dfd-4ee2-8358-97a83bd40fc8` showed normalized OpenCLIP
  image/text embeddings and a raw cosine matrix. Durable sources included
  `alimoridnejad/neural-search@00ef5a2d56a8cd5b800c362d220efd379af7b48a`
  (MIT), `AnyLoc/AnyLoc@a9fda68c55083f765b019df148a1b67614f90ee2`
  (BSD-3-Clause), `kerrj/lerf@db08d578038d884542688511bd9ad7b489a65673`
  (MIT), and
  `zai-org/Inf-DiT@e710c3f0530adc15ee92e0635164d44261cc68da`
  (Apache-2.0).
- `0768879f-03d4-4612-8572-958d1fca077a` covered precomputed normalized text
  embeddings and cosine ranking. Its remote-API/NumPy workflow and mutable
  issue provenance were rejected as implementation precedent.
- `19c1ebb4-8791-4cf5-8af4-367ccbb702d6` corroborated same-device MPS
  inference, CUDA-only autocast, and CPU-safe output. Its mutable references
  were treated only as corroboration. The broader first MPS query failed with
  GitHits HTTP 500 and produced no solution ID; the narrower retry succeeded.

Exact package-source GitHits calls:

```text
pkg_info(pypi:open_clip_torch)
search(pypi:open_clip_torch@3.3.0, encode_image/encode_text/normalize/logit_scale/softmax)
code_read(LICENSE, 1:23)
code_read(src/open_clip/model.py, 430:479)
code_read(src/open_clip/zero_shot_classifier.py, 21:68)
code_read(src/open_clip_train/zero_shot.py, 17:42)
code_read(tests/test_inference_simple.py, 25:51)
```

The pinned source establishes:

- `encode_image(..., normalize=True)` and `encode_text(..., normalize=True)`
  produce unit features in the model forward path;
- pre-softmax image logits are the dot product of normalized image and text
  features multiplied by the positive learned `exp(logit_scale)` (plus an
  optional model bias), so cosine and those logits have the same ordering for
  a fixed checkpoint;
- zero-shot text templates are encoded in bounded batches, averaged per
  class, normalized again, and concatenated;
- OpenCLIP's evaluation path ranks directly from logits; softmax is an output
  interpretation over the current candidate set, not a stable candidate
  score;
- tensors are kept on the selected device for encoding and matrix products.
  BioMiner will transfer only final normalized embedding rows to CPU through
  its existing persistent-worker boundary, which is compatible with CUDA,
  MPS, and CPU without retaining accelerator tensors in the taxonomy index.

Constraints adopted:

- encode every image batch once and reuse the normalized image embedding for
  all adaptive taxonomy stages;
- cache normalized text embeddings and use raw cosine dot products for every
  selection decision;
- make probabilities an explicitly diagnostic, candidate-set-relative API;
- fingerprint classification version, hierarchy, prompt version and stage,
  exact label text, model ID, and checkpoint before cache reuse;
- batch missing direct-mode text labels and reuse them across images;
- use a physically compact float32 index rather than Python tuples of float64
  values for the full butterfly label set;
- give species first-pass and rerank separate prompt stages and disjoint label
  sets, then rerank exactly the retained 20.

Patterns rejected:

- `(100 * image @ text.T).softmax(...)` as a rank-selection API, because adding
  or removing candidates changes every reported probability;
- comparing softmax probabilities produced from different rank candidate
  universes;
- re-embedding one crop at each adaptive rank;
- using an unvalidated label-only cache that can silently cross prompt stages,
  taxonomy revisions, or model checkpoints;
- calling the same species prompt set twice and describing the second call as
  reranking.

Morph MCP was also invoked for the required local call-site review and failed
with the exact response `Error: 429 status code (no body)`. Focused `rg` and
source reads were used for local navigation; no Morph-derived claim is made.

Pre-implementation solution IDs:

`5028e074-f077-484a-b607-cae57531390b`,
`7e71f8b6-3dfd-4ee2-8358-97a83bd40fc8`,
`0768879f-03d4-4612-8572-958d1fca077a`,
`19c1ebb4-8791-4cf5-8af4-367ccbb702d6`.

### Post-implementation verification — 2026-07-11

Strict post call:

```text
get_example(
  query="Python CLIP zero-shot classifier cache normalized text embeddings, encode each image batch once, compute raw cosine similarities for deterministic ranking, then rerank a fixed top-k with a distinct prompt ensemble without softmax selection",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `fbf0d6c3-b346-427c-8a0b-9a79caa6ae0c`. Its immutable
references included:

- `towhee-io/towhee@fe856301680713032e9613cf2500932f0ae3ad13`
  (Apache-2.0);
- `alibaba/EasyNLP@a4ee9568fa094f5825346f3acd0d65b15f1e3a95`
  (Apache-2.0);
- `huggingface/optimum-intel@6898d97242498b7dc4367a12555e4f7e52b01d16`
  (Apache-2.0);
- `lstein/PhotoMapAI@6b7ae76c6b51bed913d103e60c69a60103e712fa`
  (MIT);
- `om-ai-lab/RS5M@084cdd9596a845b69320526a7eef44b8cd3426ed`
  (MIT).

The generated example confirmed the design pattern—one normalized image
embedding, cached normalized text embeddings, raw matrix products, fixed
top-k restriction, and a distinct prompt ensemble for reranking. BioMiner did
not copy that implementation: it retains source-scoped taxonomy node IDs,
stage-specific prompt provenance, accepted GBIF mappings, deterministic
scientific-name/node-ID tie breaks, and strict cache identity checks.

The post audit re-read exact OpenCLIP 3.3.0 source at
`mlfoundations/open_clip@30573618fc375b12f094ef64cb3a1391cf611c45`
(MIT):

```text
code_read(src/open_clip/model.py, 310:365)
code_read(src/open_clip/model.py, 430:479)
code_read(src/open_clip/zero_shot_classifier.py, 35:72)
code_read(tests/test_inference_simple.py, 38:51)
```

The source signatures and shapes match the integration:

- image output is `[image_batch, embedding_dim]` and text output is
  `[text_batch, embedding_dim]`;
- `F.normalize(..., dim=-1)` makes their matrix product a cosine matrix;
- OpenCLIP logits apply the positive learned `exp(logit_scale)` after
  normalization, so unscaled cosine preserves fixed-checkpoint ordering;
- prompt batches are reshaped as
  `[class_batch, prompt_variant_count, embedding_dim]`, averaged, normalized,
  transposed, and concatenated;
- softmax occurs only after the similarity/logit matrix and is unnecessary for
  top-k selection.

Implementation comparison:

- `TaxonomyTextEmbeddingIndex.raw_similarities` returns unscaled cosine values;
  `diagnostic_probabilities` is a separately named candidate-relative helper.
- The cache uses an exact Float32 physical schema and binds classification
  version, prompt version, prompt stage, hierarchy fingerprint, label hash,
  model ID, model checkpoint, dimension, and a canonical whole-cache SHA256.
- Cache loading rejects missing labels, stage-association drift, wrong
  hierarchy/model/checkpoint, non-unit or non-finite vectors, mixed dimensions,
  and fingerprint or physical-schema changes.
- Compact immutable bytes hold the validated vector matrix instead of Python
  float objects. Prompt text is embedded once per unique stage/label.
- Prompt version `butterfly-six-rank-prompts-v4` supplies two rank-screen
  variants, two species-first-pass variants, and two disjoint species-rerank
  variants; lineage-aware rerank text uses only reviewed family/genus paths.
- `classify_path_cascade_batch` calls `embed_image_items` once for the entire
  input batch. Direct mode batches and memoizes missing text labels across the
  batch; cached mode performs no classification-time text encoding.
- Species rerank candidates are set-equal to first-pass top 20. The reranked
  top 20 retains both raw stage scores; top 5 and reported top 3 are strict
  prefixes, and the winning path follows reranked top 1.
- Existing accelerator work remains inside the persistent sidecar. Normalized
  rows cross the boundary through `detach().cpu().tolist()`, so the taxonomy
  index does not retain CUDA or MPS tensors.

Patterns rejected after comparison:

- OpenCLIP's illustrative constant `100.0` as a durable raw score;
- candidate-set softmax in cached or direct selection;
- averaging prompt vectors before preserving prompt-level audit evidence;
- persisting accelerator tensors or Python float64 tuples for every label;
- reranking a species outside first-pass top 20;
- allowing cache fallback after any identity mismatch.

Repository verification:

- focused Ruff across the staged registry, store, cache, worker, classifier,
  object runner, and their tests: passed;
- focused pytest across those components: `167 passed`;
- the first full run exposed one honest stale CLI expectation (`5` prompt rows
  versus the new staged count `12`); the assertion was updated and rerun;
- final `.venv/bin/pytest -q`: `938 passed`.

The dedicated sidecar remains absent and its installer remains insufficiently
pinned; therefore real OpenCLIP/MPS execution was not claimed or required for
this deterministic phase gate.

Post-implementation solution IDs:

`fbf0d6c3-b346-427c-8a0b-9a79caa6ae0c`.

## Phase 5 — Versioned cascade Parquet and audit semantics

### Pre-implementation verification — 2026-07-11

Dependency inspection found that `pyproject.toml` permits `polars>=1.30` and
`pyarrow>=18`, while `uv.lock` resolves Polars 1.41.2 and PyArrow 24.0.0.
Phase 5 therefore treats those locked versions—not newer registry releases—as
the API contract.

Strict call 1:

```text
get_example(
  query="Polars Python explicit nested Struct and List(Struct) schema, including typed empty DataFrame and selecting/projecting columns for stable schema evolution",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `77cf4806-5805-4e4b-b866-6fb623606bcc`. The result demonstrated
explicit `pl.Struct`/`pl.List(pl.Struct(...))`, zero-row frames constructed from
an explicit schema, and ordered projection. Its permissively licensed sources
included Polars and Apache-2.0, BSD-3-Clause, and MIT projects.

Strict call 2:

```text
get_example(
  query="PyArrow Python nested list of struct schema, typed empty table, schema projection and Parquet Zstandard output",
  language="Python",
  license_mode="strict",
  format="json"
)
```

Solution ID: `59c46397-efc0-48a6-b63e-5a7529e18947`. It confirmed the
Arrow equivalents—`list_(struct(...))`, an empty table with an explicit
schema, top-level projection, and Zstandard Parquet. A separate strict Polars
Zstandard query hit GitHits' rolling 50-example limit and returned no solution
ID; no ID or verification claim was invented for it.

Pinned source verification:

- `pola-rs/polars` tag `py-1.41.2` resolved to immutable commit
  `599a503a0997188a74750926a5cdaa47585cf8aa`; `LICENSE` confirmed MIT.
  Exact reads were
  `py-polars/tests/unit/constructors/test_constructors.py:1358-1430`,
  `py-polars/src/polars/dataframe/frame.py:4118-4205`, and
  `py-polars/tests/unit/io/test_parquet.py:115-140`.
- `apache/arrow` tag `apache-arrow-24.0.0` resolved to immutable commit
  `31b4b6c0a0a7e7c117312d285541a21446675ec6`; `LICENSE.txt:1-30`
  confirmed Apache-2.0. Focused tests at
  `python/pyarrow/tests/test_extension_type.py:733` and
  `python/pyarrow/tests/test_pandas.py:5319` exercise list-of-struct and schema
  round trips.
- GitHits `pkg_info` corroborated the package lineage. The Polars registry
  license field contains only copyright prose, so the immutable source
  `LICENSE`, not that metadata field, is the licence authority used here.

The pinned Polars source confirms that `DataFrame.write_parquet` accepts
`compression="zstd"` and uses the native writer by default; it documents Zstd
levels 1–22 with default 3. Its constructor tests prove nested
`List(Struct(...))`, inner list fields, empty lists, and exact typed zero-row
frames. The pinned Arrow tests independently confirm compatible nested schema
round trips.

Constraints adopted:

- define one ordered, versioned Polars schema as the physical contract;
- use typed `List(String)`/`List(Float64)` columns for important aligned top-k
  arrays and fixed `Struct` columns for per-rank count maps;
- reserve compact deterministic JSON only for verbose pruning trace detail;
- construct both populated and empty frames with the same explicit schema;
- project and cast in declared order before writing, rather than infer or
  merge schemas;
- write native Polars Parquet through BioMiner's existing atomic writer with
  explicit Zstandard compression and verify a physical-schema round trip;
- add nullable fields only behind an incremented classifier schema version.

Patterns rejected:

- schema inference for empty, all-null, or nested result data;
- Python-object columns or JSON blobs for first-class ranked candidates;
- permissive/diagonal schema merging and implicit column order;
- silently reinterpreting old Parquet parts under the new schema;
- switching to the PyArrow writer where the pinned Polars native writer
  already supports the required nested types and compression;
- assuming `pa.parquet` exists instead of importing `pyarrow.parquet`, or
  treating top-level `Table.select` as nested-child projection.

The local output audit found that v3 currently has no serializer. The v2
serializer sets `family_top1` from the final species path, writes overlay node
IDs into `family_top3_accepted_taxon_keys`, and retains `genus_top8`.
`RankStepResult` also discards full candidate/pruned-node detail. These are the
specific defects the Phase 5 schema must make unrepresentable.

Morph MCP was invoked for the required local call-site trace and failed with
the exact response `Error: 429 status code (no body)`. Focused source reads and
`rg` call-site discovery were used; no Morph-derived claim is made.

Pre-implementation solution IDs:

`77cf4806-5805-4e4b-b866-6fb623606bcc`,
`59c46397-efc0-48a6-b63e-5a7529e18947`.

### Post-implementation verification — 2026-07-11

A strict post-implementation `get_example` call for a versioned Polars
classifier schema, fixed Struct counts, typed empty frames, deterministic JSON
traces, and Zstandard round trips returned the exact GitHits error:

```text
You have reached the limit of 50 generated examples in a rolling 24-hour
window. Please try again in about 67161 seconds.
```

It produced no solution ID, and none is claimed. GitHits package/source
navigation remained available, so post-verification continued against the
same immutable dependency releases behind the two successful strict Phase 5
solutions rather than treating the generated-example quota as dependency API
failure.

Exact post reads:

```text
code_read(
  pola-rs/polars@599a503a0997188a74750926a5cdaa47585cf8aa,
  py-polars/src/polars/dataframe/frame.py,
  4118:4205
)
code_read(
  pola-rs/polars@599a503a0997188a74750926a5cdaa47585cf8aa,
  py-polars/tests/unit/constructors/test_constructors.py,
  1358:1430
)
code_read(
  pola-rs/polars@599a503a0997188a74750926a5cdaa47585cf8aa,
  py-polars/tests/unit/io/test_parquet.py,
  115:140
)
code_read(
  apache/arrow@31b4b6c0a0a7e7c117312d285541a21446675ec6,
  python/pyarrow/tests/parquet/test_parquet_writer.py,
  1:80
)
```

The post reads reconfirmed the exact APIs now exercised by BioMiner:

- Polars 1.41.2 declares native `DataFrame.write_parquet` with default
  `compression="zstd"`, supports explicit compression levels, and uses
  `use_pyarrow=False` by default;
- its constructor tests build nullable `List(Struct(...))` data, nested inner
  lists including empty lists, and an exact zero-row frame from the same
  explicit schema;
- its Parquet tests write/read list columns from buffers across native and
  PyArrow engines and assert frame equality;
- Arrow 24.0.0 imports `pyarrow.parquet as pq`, creates a writer from an exact
  schema, writes repeated tables, closes it, and reads the result back.

Implementation audit:

- The post-audit found three contract holes: the physical-schema validator
  dropped null classifier versions before its uniqueness check; row
  normalisation silently discarded unknown legacy fields such as
  `genus_top8`; and null or unsupported pruning-trace versions could pass.
  Validation now rejects all three, with regressions alongside legacy-version
  and physical-dtype drift cases.
- `PATH_CASCADE_OUTPUT_SCHEMA_VERSION` establishes a new physical contract;
  classification-v2 rows are not reinterpreted as it.
- Important top-k names, reviewed node IDs, GBIF species keys, and raw scores
  are aligned typed list columns. A single vectorised validation pass rejects
  null, misaligned, or oversized candidate arrays. No non-species accepted-key
  column exists in the new cascade schema.
- Four fixed six-field Struct columns hold candidate, retained, and active-path
  counts. Typed empty frames and reviewed skipped SUBTRIBE rows round-trip with
  the exact schema under Zstandard.
- Rank-stage top1 fields come only from the rank step. `selected_*` fields come
  only from the reranked winning species path. A regression proves these can
  legitimately differ.
- Every screen stage stores the full sorted union IDs and raw scores, retained
  IDs, pruned IDs, parents, counts, path counts, reviewed-skip count, and skip
  reason. The compact deterministic trace contains six rank/first-pass steps
  plus the distinct species rerank and contains no cumulative score.
- Object evidence projection retains the new fields. Photo summaries expose
  winning family and genus and compare accepted species keys before name
  aliases when detecting multi-object disagreement.
- Review and QA accept a reviewed missing SUBTRIBE, reject an unexplained one,
  and no longer apply the invalid v2 rule that every global top20 species must
  lie under the final winning family.
- Evaluation uses family names for v3 rank/selected metrics instead of
  comparing reviewed overlay IDs with GBIF family keys. Raw cosines are not
  converted into a fake entropy diagnostic. The genus metric is top3, with a
  temporary read fallback for historical top8 rows.

Patterns rejected after comparison:

- a durable `List(Struct)` trace whose evolution would force nested schema
  migration; first-class arrays plus compact JSON are clearer here;
- inferred schema for empty or skipped-rank output;
- nullable list values where `[]` has a precise audit meaning;
- overlay IDs in any GBIF accepted-key list;
- fallback from missing v3 selected rank to rank-stage top1;
- candidate-set probability assumptions for raw cosine scores;
- diagonal merging or silent reinterpretation of historical Parquet parts.

Repository verification:

- focused Ruff over schema, classifier, object propagation, review, QA,
  calibration, evaluation, and tests: passed;
- broad focused pytest including cloud work and production skeleton:
  `302 passed`;
- final `.venv/bin/pytest -q`: `953 passed`.

Post-implementation solution IDs used for the immutable-source comparison:

`77cf4806-5805-4e4b-b866-6fb623606bcc`,
`59c46397-efc0-48a6-b63e-5a7529e18947`.

## Phase 6 — Production work identity and retry-safe cascade wiring

### Pre-implementation verification — 2026-07-11

Research questions:

- Which output-affecting settings must participate in a deterministic cascade
  work key?
- How should desired-computation identity remain distinct from serialized
  output checksums and queue attempt identity?
- What queue contract remains safe under worker loss, redelivery, and stale
  claims?
- Which rank and taxonomy settings must be explicit rather than inherited from
  ambient production defaults?

Repository dependency evidence:

- `pyproject.toml` requires Python 3.14 or newer.
- `uv.lock` pins Polars 1.41.2, PyArrow 24.0.0, DuckDB 1.5.4,
  HTTPX 0.28.1, and Pydantic 2.13.4. Phase 6 work-key hashing uses Python's
  standard JSON and SHA-256 APIs; the dependency inventory was inspected
  before querying external implementation patterns.

GitHits calls and results:

1. Two delegated strict-licence `get_example` calls for deterministic
   content-addressed work keys, immutable batch identity, retry-safe queues,
   and configuration fingerprints hit the rolling 50-example quota. The
   exact retry estimates were 66513 and 66465 seconds. A concurrent main-agent
   call already in flight returned the same error with 66402 seconds. These
   calls issued no solution ID, and none is claimed.
2. The earlier strict solution
   `a04caaf5-9c66-4bcf-9e7b-8d08dbe3a062` was re-used only for its directly
   relevant canonical-adjacency/content-hash evidence. Its immutable MIT and
   Apache-2.0 sources remain recorded in Phase 1; it is not presented as new
   queue evidence.
3. `code_read` inspected
   `iterative/dvc@f74c1c0e709de61f571905802bc0c75035dc6ef2`:
   `dvc/utils/__init__.py:20-51`, `dvc/stage/cache.py:20-80`, and
   `LICENSE:1-5`. DVC is Apache-2.0. The source recursively normalizes a
   mapping, serializes it with sorted JSON keys, hashes UTF-8 bytes with
   SHA-256, and distinguishes a stage-input key from an output-content value.
4. `code_read` inspected
   `celery/celery@1432d9b6c6868a77e7ee2ede1650da00a8d187ac`:
   `celery/worker/request.py:670-705`,
   `t/unit/worker/test_request.py:398-450`, and `LICENSE:1-38`. Celery is
   BSD-3-Clause. Its late-acknowledgement path stores the result before
   acknowledgement, and its worker-loss tests verify that requeued work is
   redelivered without a premature terminal failure.
5. Morph MCP was called first for the required local production call-site
   trace and failed with the exact response
   `Error: 429 status code (no body)`. Focused `rg` and source reads identified
   the local paths; no Morph-derived claim is made.

Constraints adopted:

- Canonicalize the complete output-affecting settings mapping with sorted
  keys, compact separators, UTF-8, an explicit contract version, and SHA-256.
- Preserve semantic list order. The six-rank order is identity-bearing; any
  unordered collection must be sorted before hashing.
- Include beam strategy and width, ordered ranks, classification and prompt
  versions, taxonomy and embedding-cache fingerprints, species first-pass,
  rerank and report widths, and the rerank prompt version.
- Keep the stable computation work key, batch/part identity, durable output
  checksum, and mutable queue attempt/claim identity separate.
- Assume at-least-once execution: publish durable output before completion,
  reclaim expired claims, and make repeated publication checksum-idempotent.
- A retry of the same immutable computation retains the same work key. Any
  output-affecting configuration change produces a new work key.

Patterns explicitly rejected:

- Python `hash()`, `repr`, pickle, timestamps, attempt numbers, and random IDs
  as durable work identity.
- Excluding a setting merely because it currently equals a default; ambient
  defaults change and must not reinterpret old work.
- Mutating the mapping while deriving its key, as DVC's specialized
  `key=True` transformation does for its own stage representation.
- Claiming exactly-once delivery, acknowledging before durable publication,
  marking worker loss terminal, or allowing a stale claimant to overwrite a
  committed shard.
- Treating object existence alone as successful concurrent publication; an
  existing object must have the expected immutable checksum and identity.

Effect on BioMiner's design:

- The public family-only width is replaced by one fixed global rank beam.
- Production requests and manifests carry the full cascade contract instead
  of reconstructing it from scattered defaults.
- Rolling and cloud work keys will use one canonical settings payload and
  include taxonomy/cache fingerprints so stale v2 or stale-cache work cannot
  collide with classification-v3 output.
- Queue retry/claim fields will not participate in the computation key.

Pre-implementation solution ID used for the canonical-hash comparison:

`a04caaf5-9c66-4bcf-9e7b-8d08dbe3a062`.

### Post-implementation verification — 2026-07-11

Two strict-licence post-implementation `get_example` calls for canonical work
identity, retry-safe publication, and redelivery-safe queue handling hit the
rolling 50-example quota. The exact retry estimates were 63771 and 63750
seconds. Neither call produced a solution ID, and none is claimed.

Post-verification therefore used immutable source reads from the same licensed
projects inspected before implementation:

```text
code_read(
  iterative/dvc@f74c1c0e709de61f571905802bc0c75035dc6ef2,
  dvc/stage/cache.py,
  150:195
)
code_read(
  celery/celery@1432d9b6c6868a77e7ee2ede1650da00a8d187ac,
  celery/worker/request.py,
  677:696
)
code_read(
  celery/celery@1432d9b6c6868a77e7ee2ede1650da00a8d187ac,
  t/unit/worker/test_request.py,
  398:450
)
```

DVC is Apache-2.0. Its stage cache keeps the deterministic input-derived key
separate from the output value, detects an already-identical cached entry, and
publishes through a temporary location followed by a move. Celery is
BSD-3-Clause. Its late-acknowledgement path stores a successful result before
acknowledgement, while worker-loss tests verify requeue and redelivery without
premature terminal failure. These are implementation-pattern comparisons, not
production dependencies.

Morph MCP was invoked again for the required post-implementation call-site
audit and failed with the exact response `Error: 429 status code (no body)`.
Focused `rg`, source reads, and executable tests supplied the local evidence;
no Morph-derived claim is made.

Implementation audit:

- One fixed production cascade contract now owns global current-rank top 3,
  the ordered six ranks, species first-pass top 20, distinct rerank top 5, and
  report top 3. Production CLI width switches were removed.
- Hierarchical production requires classification-v3 artifacts and a validated
  model-matched taxonomy text-embedding cache. Missing, stale, v2, or
  fingerprint-mismatched inputs fail explicitly; there is no direct-prompt or
  v2 fallback.
- Each worker or stage loads one taxonomy index, embeds each crop once per
  batch, and uses cached dot products for every rank and species prompt.
- Work identity includes the beam strategy and width, ordered ranks,
  classification and prompt versions, taxonomy and hierarchy fingerprints,
  embedding-cache fingerprint, all three species widths, and rerank prompt
  version. Workers recompute this immutable identity before work. Attempt,
  lease, and retry metadata are deliberately excluded.
- The rolling score stage previously wrapped the canonical BioCLIP work key in
  an ad-hoc `:score:` key. That defect was removed; rolling and cloud now use
  the same canonical computation key and preserve retry identity.
- Local and cloud fake-input regressions serialize equivalent v3 rows. The
  rolling worker remains the sole cloud production visual route; the direct
  local path is explicitly development-only.
- The first full-suite gate found three untested developer plumbing failures:
  the benchmark still loaded `FiveRankTaxonomyStore` and passed the removed
  `taxonomy_store=` argument. The benchmark now builds validated v3 artifacts,
  records reviewed SUBTRIBE skips, constructs one in-memory fake-model cache,
  and asserts one image embedding per eligible crop.
- The follow-up call-site audit found the same stale contract in the live M5
  Pro benchmark. It now requires `--taxonomy-text-embedding-cache`, validates
  the cache against the exact BioCLIP model name and checkpoint before detector
  work, records all taxonomy/cache fingerprints, and no longer exposes the
  obsolete family/species width flags.

Patterns rejected after comparison:

- ambient defaults, partial hashes, Python `hash()`, or attempt numbers in a
  durable computation key;
- acknowledging completion before durable publication or claiming
  exactly-once queue execution;
- treating object existence as success without identity and checksum
  agreement;
- family-only or independent per-rank production widths;
- classification-v2 reuse, stale-cache reuse, and direct text-prompt fallback;
- model-free benchmark metrics that fabricate the retired six-score-call path
  instead of measuring the one-image-embedding v3 path.

Repository verification:

- the first full gate reported the stale benchmark contract exactly as
  `3 failed, 992 passed`; the failures were fixed rather than hidden;
- final repository-wide `uv run ruff check .`: passed;
- expanded rolling, cloud, orchestrator, CLI, production, plumbing, and live
  benchmark focused suite: `207 passed`;
- final `.venv/bin/pytest -q`: `996 passed`.

Post-implementation solution ID reused only for the canonical-hash comparison:

`a04caaf5-9c66-4bcf-9e7b-8d08dbe3a062`.

## Phase 7 — Deterministic cascade acceptance benchmarks

### Pre-implementation verification — 2026-07-11

Research questions:

- How should a model-free classifier benchmark make ranking regressions
  mutation-sensitive and deterministic?
- Which top-k invariants expose false success when the candidate universe is
  too small or ties are left implicit?
- Which per-stage candidate, retention, work, and timing measurements are
  useful without turning wall-clock values into correctness assertions?
- How should historical and current beam algorithms be compared without
  restoring the historical classifier to production?

Strict GitHits calls:

1. A delegated strict-licence query for deterministic benchmark rankings and
   top-k recall hit the rolling generated-example quota with an exact retry
   estimate of 62800 seconds.
2. A delegated strict-licence query for stage timing and candidate-count
   telemetry hit the same quota with an exact retry estimate of 62783 seconds.
3. A main-agent strict-licence query combining classifier benchmarks, ranking
   invariants, top-k recall, and per-stage telemetry hit the quota with an
   exact retry estimate of 62620 seconds.

No call returned a solution ID, and no new Phase 7 solution ID is claimed.
The directly relevant Phase 3 strict solutions remain reusable evidence for
global-frontier and stable-order behavior:

- `ff07eee2-f883-4dce-a4cd-37c584518675`;
- `01e56d9b-226c-474c-8d7e-85891e70c457`.

GitHits immutable source verification then inspected
`scikit-learn/scikit-learn@6b9e392862ac86f6a3f3b71ee89622d5af49bb4e`.
`COPYING:1-29` establishes BSD-3-Clause. Exact source/test windows were:

```text
code_read(sklearn/metrics/_ranking.py, 2138:2248)
code_read(sklearn/metrics/tests/test_ranking.py, 2121:2231)
code_read(sklearn/model_selection/_search_successive_halving.py, 335:405)
code_read(examples/miscellaneous/plot_outlier_detection_bench.py, 87:96)
```

The source establishes these patterns:

- top-k tie behavior is documented rather than inherited accidentally;
- labels are required to be unique and ordered before scoring;
- a stable sort is used before slicing the top-k candidates;
- `k >= candidate_count` is identified as a meaningless perfect result;
- fixed score matrices assert exact top-1, top-2, and top-3 outcomes;
- a seeded fixture checks that top-k success is monotonic as k grows;
- an explicit equal-score fixture asserts the documented tie result;
- successive-halving telemetry stores candidates entering each iteration,
  iteration/resource metadata, and candidates retained after pruning;
- elapsed duration is bracketed with `perf_counter()` around the operation
  being measured.

The source's exact numeric-index tie policy is not BioMiner's identity policy.
BioMiner will retain its explicit taxonomic secondary keys: scientific name
and stable node ID. Scikit-learn is evidence for documenting and testing a
tie policy, not a dependency or an algorithm to copy.

Constraints adopted:

- Use fixed synthetic scores and exact expected node orders. Do not load a
  real model, network resource, or biological taxonomy in acceptance tests.
- Validate a seven-family fixture through the real classification-v3 frame,
  QA, fingerprint, and `PathTaxonomyStore` contracts.
- Record candidate count, retained actual-node count, active paths before and
  after, unique labels scored, reviewed skip count, and elapsed duration for
  every rank/prompt stage.
- Treat timing as non-negative telemetry only. Exact durations and throughput
  are not test invariants.
- Make top-k monotonicity non-decreasing, because plateaus are valid.
- Ensure selected benchmark branches contain more than 20 species so top-20
  assertions cannot pass trivially.
- Keep the historical cumulative selector private to benchmark/test code and
  assert exact differing subfamily IDs against the production result.

Patterns explicitly rejected:

- using a trained estimator or real BioCLIP weights in the core benchmark;
- treating a perfect result with `k >= candidate_count` as useful evidence;
- relying on input order or an undocumented numeric index for ties;
- asserting exact wall time or minimum throughput in unit tests;
- reducing active-path count to beam width, because one retained taxon node
  can legitimately represent many species paths;
- counting reviewed skipped SUBTRIBE paths as retained beam nodes;
- placing historical cumulative pruning in `src/biominer/bioclip/` or calling
  the legacy five-rank classifier.

Morph MCP was invoked for the required benchmark/call-site discovery and
failed with the exact response `Error: 429 status code (no body)`. Focused
local reads and `rg` supplied the repository evidence; no Morph-derived claim
is made.

Pre-implementation solution IDs reused from the earlier strict ranking
verification:

`ff07eee2-f883-4dce-a4cd-37c584518675`,
`01e56d9b-226c-474c-8d7e-85891e70c457`.
