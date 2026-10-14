# Validation — 2026-10-15

A fresh Python 3.11 environment installed TensorFlow 2.15.1, Flower 1.13.1,
NumPy 1.26.4 and scikit-learn 1.6.1 on macOS ARM64. CPU validation:

- `python -m pytest -q`: 25 tests passed, including four two-round, two-client
  training runs for FedAvg, NNC, AKD and NA-FedAvg through standard Flower
  protobuf serialization. The AKD run activates distillation, preserves its
  client-local projection and aggregates classifier weights in both rounds.
- NA-FedAvg includes all clients during warmup and partial participation later;
  the test also covers an explicitly skipped zero-impact subset update.
- Numerical fixtures verify KNN class identities, image scaling, seed forwarding,
  energy weights with noncontiguous identities, distillation without model
  regularizers, fixed-origin cutout and confident-learning filtering.
- Separate two-round CPU smokes passed for label smoothing, bi-tempered loss
  and confident learning (including cached pruning on its second round).
- `pip check` found no broken requirements; source compilation passed.
- Earlier launcher/cache/data regressions remain in the suite. Missing-class
  KNN and energy identity fixtures failed before their repairs.

The full paper datasets, CLIP/TRILLsson extraction, distributed Ray/GPU training
and published accuracy tables were not reproduced. Synthetic-run accuracy is
not a performance estimate. The configured Linux CI has not been observed
remotely as part of this local work.

## Small participation regression

A focused regression now constructs the actual server/strategy and checks that
`participation=0.01` with two available clients schedules one fit instruction.
Before the minimum-client floor, it scheduled zero. FedLN covers both its base
server and NA-FedAvg override; FedCompress covers its compressed strategy.
These focused checks passed after the repair.

Recovered audio: three focused tests passed in 38.89 s, including generated
TFRecord loading and actual two-round NNC/AKD CNN training. Details in
[RECOVERED_DATA.md](RECOVERED_DATA.md).
