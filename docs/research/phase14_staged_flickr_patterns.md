# Phase 14 staged Flickr inference patterns

Task 14.4.3 used GitHits solution
`c2f3029a-b5af-4931-af25-11b39b7edfdc` to verify open-source
implementation patterns before local implementation. The returned example was
distilled from permissively licensed projects including
[`0-5788719150923125/praxis`](https://github.com/0-5788719150923125/praxis/blob/cdcf65b40f1576cd4fb60965d00481cf03cc7a32/praxis/trainers/mono_forward/trainer.py)
(MIT),
[`SimonBartosDev/opencure`](https://github.com/SimonBartosDev/opencure/blob/e2798c5770ac7841a536e9ab618b035ebd984665/scripts/modal_app.py)
(Apache-2.0), and
[`NVIDIA/physicsnemo`](https://github.com/NVIDIA/physicsnemo/blob/1cc227e69be1d4c7b7b1cd7aa175f4275d34e878/examples/cfd/external_aerodynamics/globe/drivaer/train.py)
(Apache-2.0).

BioMiner-native decisions extracted from those patterns are:

- cumulative P1/P2/P3 gates use deterministic record limits;
- SQLite stores resumable operational status while Parquet remains the durable
  analytical output;
- one BioCLIP process and one YOLOE process are reused across bounded batches;
- a failed batch is bisected so one bad record remains retryable instead of
  becoming a biological negative;
- gates validate finite scores, record coverage, failure rate, throughput, and
  memory instrumentation;
- complete checkpoints return without reloading either model;
- all Flickr image files are temporary, content-addressed, and deleted after
  each classified batch.

No external implementation was copied wholesale. The implementation preserves
BioMiner's target-always-scored, no-pruning, full-frame, route-separation, and
experimental-screening semantics.
