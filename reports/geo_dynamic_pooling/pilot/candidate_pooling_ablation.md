# Family/geography candidate ablation

Status: deterministic fixture execution completed.

The frozen seven-case register was built through BioMiner's production
family/geography candidate-union validator, then scheduled independently by
all three production policies. Each case contained the same five accepted
Papilionidae taxa. All 21 case-strategy results retained the target and the
complete candidate union; only evaluation order changed.

| Strategy | Mean target rank | Max rank | Target at 1 | Target at 3 | Target at 5 |
|---|---:|---:|---:|---:|---:|
| geography first | 1.000 | 1 | 7/7 | 7/7 | 7/7 |
| family first safe | 2.714 | 3 | 1/7 | 7/7 | 7/7 |
| parallel family/geography union | 1.000 | 1 | 7/7 | 7/7 | 7/7 |

These values describe the position of the fixture-declared target in a
complete candidate schedule. They are not classification accuracy, reviewed
recall, biological performance, or evidence that one strategy is universally
better. The deliberately varied fixture family ordering stresses the safety
contract; geography-first and parallel scheduling explicitly prioritize the
target/safety union, while family-first safe may evaluate family-priority rows
first but still retains the target by cutoff three.

The no-geography case contains no local geographic scopes, scores, or support
counts. It falls back to non-geographic scheduling without converting missing
source geography into absence.

Wall-clock timing was not instrumented in this structural ablation. No
candidate strategy is selected, production-default eligibility is false, and
occurrence release remains unauthorized.
