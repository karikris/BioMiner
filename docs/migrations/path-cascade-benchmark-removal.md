# Path-cascade benchmark removal

Date: 2026-07-19

The `biominer dev vision benchmark-cascade` command and its synthetic taxonomy
fixture were removed during the post-uplift simplification. They exercised the
legacy hierarchical path cascade without loading a model or reviewed labels;
their output was implementation-plumbing evidence and was not consumed by the
adaptive geography-conditioned pooling workflow.

Historical benchmark artifacts, verification reports, and Git revisions remain
unchanged. They must not be presented as model accuracy, biological evidence,
or a production readiness gate. Current throughput checks remain available via
the bounded plumbing benchmarks, while scientific evaluation continues to
require source-bound reviewed labels.
