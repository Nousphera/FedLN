# Reproducibility and method correspondence

The maintained entry point is `python run.py`. It invokes the original method
clients and server strategies with TensorFlow 2.15.1 and Flower 1.13.1. Clients
train sequentially on one host; every fit request and response passes through
stock Flower protobuf serialization. This is a computation/transport check,
not a distributed timing benchmark. The old `src/run.sh` Ray entry points and
`src/requirements.txt` remain for historical inspection. Their distributed
execution is not qualified with the maintained environment.

## Setup and bounded checks

Use Python 3.11 in an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
python run.py --method akd --dataset_name synthetic --num_clients 2 \
  --num_rounds 2 --train_epochs 1 --batch_size 8 --knn_neighbors 5 \
  --output runs/akd-smoke
```

The generated fixture has 32 training and eight test images with two classes,
and deterministic 512-dimensional features derived from the fixture labels.
It is not CIFAR, CLIP, or evidence of method accuracy. Runs write configuration,
runtime versions, round losses/accuracies, sample counts and client metrics to
`results.json`, and global arrays to `final-parameters.npz`. Use a new output
directory for every invocation. Temporary per-client caches are owned and
cleaned by that invocation. A failed process may leave its own cache directory.

The default device is CPU. `--device cuda` uses the original GPUtil selection
and needs a working NVIDIA setup; GPU and multi-host runs are not validated.
`--method-help` displays method-specific underscore-style arguments.

## Paper experiments and available inputs

The final TIST article describes three **alternative** methods: NNC corrects
labels before local training; AKD applies MAE feature supervision to clients
whose estimated noise exceeds 0.15; NA-FedAvg estimates energy scores once and
then changes aggregation weights. This repository supplies CIFAR-10 and
EuroSAT. It does not supply the final paper's full audio, human-noise,
FashionMNIST or PathMNIST experimental pipelines.

A configuration matching the main paper's exposed CIFAR-10 settings is:

```bash
python run.py --method nnc --dataset_name cifar10 --model_name resnet20 \
  --num_clients 30 --num_rounds 200 --participation_rate .8 \
  --train_epochs 1 --batch_size 128 --lr .001 --knn_neighbors 100 \
  --noisy_frac .8 --noise_level .4 --noise_sparsity .7 --seed 42 \
  --output runs/cifar10-nnc-seed42
```

Change `nnc` to `akd` for feature distillation. NA-FedAvg uses
`--method na_fedavg --est_noise_round 30 --percentile 75` and does not accept
`--knn_neighbors`. The paper studies multiple noise levels/sparsities and
three trials; one command does not reproduce its tables. The supplied random
sharding also differs from the paper's explicitly controlled 25% quantity
variance. These commands exercise the released subset, not all paper settings.

CIFAR-10 downloads through Keras. Its included embedding array has 50,000 rows
and 512 float16 columns; indices refer to the Keras training order. EuroSAT
requires `eurosat/eurosat.npz` below `TFDS_DATA_DIR` (or `src/data`), with
`images` and `labels`. The original first 21,600 rows train and the remaining
5,400 test. Its embeddings have 27,000 rows. The original EuroSAT archive/order
and embedding extraction scripts are unavailable: a newly shuffled dataset
cannot be assumed to align. No embedding files were changed by maintenance.
Record dataset/embedding hashes, seeds and hardware before a scientific run.

## Correctness repairs and remaining source conventions

The original baseline remains at `federatedml-baseline`; the pre-repair audit is
[HISTORICAL_AUDIT.md](HISTORICAL_AUDIT.md). Maintenance changes affect numerical
outcomes and should be reported when comparing with published tables:

- KNN keeps scikit-learn's fitted class IDs, including shards missing classes.
- CLI seed reaches all repeated data loads and shard construction; AKD uses the
  same labels, order, batch size and noise settings as its original client.
- uint8 image normalization scales once instead of twice. TensorFlow's native
  GroupNormalization replaces Addons. The original fixed-origin cutout is
  preserved: for the CIFAR mask size 16, the visible 8×8 top-left area is zero.
  It is not a random crop location. See [Addons' original implementation](https://github.com/tensorflow/addons/blob/v0.23.0/tensorflow_addons/image/cutout_ops.py).
- AKD transmits only classifier weights. Its private projection is cached by
  client identity and reused in later rounds; embedding dimensions are checked.
  Clients first participating after the activation round still estimate noise.
- Server evaluation includes categorical cross-entropy. Distillation supports
  models without regularization and evaluates only classification outputs.
- Confident-learning filtering now removes every selected example, including
  when restoring a cached mask; it no longer filters batches as examples.
- NA-FedAvg handles client IDs as identities and requires all warmup scores.

**NA-FedAvg ambiguity is retained explicitly:** the final article says noisy
instances are below the energy threshold, but defines a step function for
values above it. Recovered code uses `local_energy >= percentile(global_energy)`
and, for affected clients, `min(0.1*n, n-noisy)` as impact. The paper's
aggregation description instead gives `(1-noise_fraction)*n`. This release
keeps the recovered convention rather than declaring either reading an exact
paper reproduction. Zero total impact raises a clear error instead of dividing
by zero. If a later participating subset has zero total impact, its update is
skipped explicitly and the previous global model is retained. This limitation must be resolved with the original experiment records
before quantitative reproduction claims.

AKD includes embedding supervision with MAE and coefficient 10; its stored
historical temperature does not enter the MAE objective. Logit-teacher AKD,
embedding extraction and full paper datasets are outside the supplied source.
