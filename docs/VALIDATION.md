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
TensorFlow 2.15.1 and NumPy 1.26.4. Initial CLIP extraction used Python 3.12,
PyTorch 2.8.0 and Transformers 4.57.3; the optional environment was subsequently
updated and revalidated as described below. Actual TRILLsson used TensorFlow
2.15.1. Training `pip check`, compilation and whitespace
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

### Current CLIP environment check

The same 32-image fixture and pinned CLIP model revision were extracted with
Python 3.11.15, PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.19.0,
NumPy 2.3.5 and Pillow 11.3.0. These were the optional environment's pins for this check;
the maintained requirement now uses Pillow 12.3.0 (see the follow-up below).
Transformers 5 defaults to torchvision preprocessing; explicitly selecting the
PIL backend preserves the earlier maintained processor's pixels. The extractor
records its actual processor class/backend. With this selection, the 32×512
embeddings were finite and matched the earlier output within `rtol=1e-4,
atol=1e-5`; maximum absolute difference was `1.8597e-5`, mean absolute difference
`7.4312e-7`, and maximum per-image relative L2 difference `3.3004e-6`. Outputs
were not bitwise identical. This is a bounded compatibility check, not evidence
of equivalent benchmark results. The TensorFlow training implementation did not
change, so its previously passing 43-test suite was not rerun for this follow-up.


### Pillow 12.3 follow-up

The maintained CLIP requirements now pin Pillow 12.3.0, which includes the fixes
in [Pillow's release notes](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html).
The same cached model revision and 32-image input were extracted again with the
PIL processor and unchanged PyTorch/Transformers/NumPy versions. All 32×512
features were finite, IDs matched, and the features were bitwise equal to the
Pillow 11.3 output (maximum absolute difference zero). This checks this fixture's
compatibility; it does not establish benchmark equivalence.

## Setup and native execution follow-up

The final Python 3.11 CPU suite passed **52 tests**. Added checks cover
prepared NPZ import/validation, wrong image shape/labels, no overwrite, content
tampering, safe tar extraction and data-loader consumption. Public corpora were
not downloaded in this check; generated source NPZ files were prepared with the
public CLI and consumed by actual training. No benchmark result is inferred.

Real `flwr run .` local simulations completed using pinned Flower 1.13.1 and
Ray 2.10.0, with real client updates and saved final arrays/results. Tests also
cover stable logical-ID sampling despite changed transport IDs and effective
post-evaluation checkpoint weights. See [FLOWER_APP.md](FLOWER_APP.md).
