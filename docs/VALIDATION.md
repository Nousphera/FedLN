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

Full dataset benchmarks, distributed Ray/GPU training and published accuracy
tables were not reproduced. Actual CLIP/TRILLsson extraction was subsequently
validated on generated inputs as recorded below. Synthetic-run accuracy is
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

## Main-paper paths and authentic teachers

Final complete suite: **43 tests passed in 119.53 s**. Training used Python 3.11,
TensorFlow 2.15.1 and NumPy 1.26.4. Actual CLIP extraction used Python 3.12,
PyTorch 2.8.0, Transformers 4.57.3, NumPy 2.5.4 and Pillow 12.3.0; actual
TRILLsson used TensorFlow 2.15.1. Training `pip check`, compilation and whitespace
checks passed. The configured remote CI has not been observed for this commit.

The maintained additions validate FashionMNIST, PathMNIST and human-noise CIFAR
training using generated canonical NPZ inputs. Tests cover fixed class counts,
stable row identities, human training labels with clean evaluation, optimizer
defaults, rejected record/sidecar reordering, rejected feature/source mismatch,
forward student-to-global KL, and frozen teacher weights. Both AKD variants run
two actual federated rounds with distillation active in the generated fixture.

Authentic teacher checks used CLIP ViT-B/32 at the pinned revision on 32 generated
images (32×512 features), and the official TRILLsson3/1 SavedModel on eight
generated waveforms (8×1024). CLIP features fed two actual NNC rounds; TRILLsson
features fed two actual AKD rounds; global model parameters changed each round.
The clean audio fixture did not activate feature distillation; separate noisy
fixtures cover activation. CIFAR-10N and CIFAR-100N official annotation files
both converted successfully with restricted loading, each with 50,000 rows.

TFDS 4.9.4's actual SpeechCommands0.0.3 builder and twelve-class mapping were
inspected. Its mocked-data API exercised the exporter entry point for all three
splits without downloading the full corpus. A second check ran generated
waveforms through the maintained exporter, real TRILLsson, and CNN AKD.
See [PAPER_DATA.md](PAPER_DATA.md) for exact teacher identities and the maintained
recipe boundaries. These checks are execution evidence, not accuracy estimates.
