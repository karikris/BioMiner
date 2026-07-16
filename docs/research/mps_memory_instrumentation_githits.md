# MPS memory instrumentation patterns

GitHits solution `7fbd1523-65ec-48cd-94e4-687934d81fe0` was used to verify
open-source PyTorch MPS memory-instrumentation practice before implementation.

The BioMiner-native implementation samples the public `torch.mps` allocator
APIs:

- `current_allocated_memory()` for tensor allocation owned by PyTorch;
- `driver_allocated_memory()` for total Metal driver allocation;
- `recommended_max_memory()` for the device's recommended working-set limit.

The persistent BioCLIP worker reports the three sampled counters with its model
attestation and retains invocation-local peaks for the two changing allocation
counters. Non-MPS devices report `not_applicable`; unavailable or failing APIs
report `not_instrumented`. Reports never guess unsupported values.

A real five-image Apple MPS smoke run verified all counters through the normal
workflow report. No external implementation was copied wholesale.
