# Papilio demoleus adaptive initial scoring

The current adaptive workflow was executed with production APIs and fixture
evidence. It reached provisional scoring with zero prior reference reviews.
The measured fixture baseline was 502.480375 ms to first provisional score and
0.448491 MiB peak traced memory.

A live rerun was not performed. The earlier prototype manifests remain in the
repository, but their ignored local artifacts are absent from this checkout.
Those manifests record 81 provider-supported references, 81 embeddings, 26
prototypes and 13,496 classified Flickr records under an earlier prototype-only
contract; none of those counts is represented as a current adaptive execution.

The unexecuted live work is explicit: acquire a current-policy durable GBIF
bank, generate its BioCLIP embeddings, build route-separated support-train
prototypes, and score the current Flickr workload with complete current-policy
resource provenance.

Provider support remains provisional, raw scores are not probabilities, and
neither fixture evidence nor provisional scoring authorizes scientific release.
