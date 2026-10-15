# Methods and configuration

[Setup](../README.md#installation), [data preparation](DATA_SETUP.md), and
[teacher extraction](PAPER_DATA.md) configure the inputs. The native Flower
App and sequential `run.py` use the same clients, models, and server strategies.

## Training methods

| Method | Operation |
| --- | --- |
| `nnc` | Correct local labels with nearest neighbors in pretrained feature space before training. |
| `akd` | Estimate local label disagreement and add distillation for clients above the configured threshold. |
| `na_fedavg` | Collect energy scores from every client once, then adjust aggregation weights. |
| `fedavg` | Aggregate local classifiers using sample counts. |
| `bi_tempered_loss` | Train with bi-tempered classification loss. |
| `label_smoothing` | Train with smoothed target distributions. |
| `confidence_learning` | Filter selected noisy examples before training. |

NNC, AKD, and NA-FedAvg are alternative interventions. NNC and AKD require
teacher features aligned to the training data; FedAvg and NA-FedAvg use labels
and model outputs directly. KNN retains the class IDs fitted on each client,
including clients whose local data omits classes.

## Data, partitions, and optimization

The seed controls repeated data loads, synthetic label noise, and client shards.
Vision data uses seeded random nonoverlapping assignment. Speech Commands
assigns complete prepared-record shards to clients deterministically; provide
at least as many shards as clients. These partitioners allow client sample
counts to vary naturally. The paper's controlled 25% quantity variation is a
separate experimental setting.

Image normalization scales uint8 inputs once. CIFAR cutout uses a fixed origin:
a mask size of 16 zeros the visible top-left 8×8 area. See
[TensorFlow Addons' cutout definition](https://github.com/tensorflow/addons/blob/v0.23.0/tensorflow_addons/image/cutout_ops.py).
The model uses TensorFlow's GroupNormalization. Optimizer defaults and teacher
preprocessing are specified in [PAPER_DATA.md](PAPER_DATA.md).

For prepared CIFAR-10 and aligned features:

```bash
python run.py --method nnc --dataset_name cifar10 --data_root data \
  --embeddings_dir "$PWD/features" --model_name resnet20 \
  --num_clients 30 --num_rounds 200 --participation_rate .8 \
  --train_epochs 1 --batch_size 128 --optimizer sgd --lr .1 --momentum .9 \
  --knn_neighbors 100 --noisy_frac .8 --noise_level .4 --noise_sparsity .7 \
  --seed 42 --output runs/cifar10-nnc-seed42
```

Choose and record the noise level, sparsity, client participation, and seed for
each experiment. `--method-help` lists the selected method's arguments. CPU is
the default; `--device cuda` selects an NVIDIA device through GPUtil.

## AKD objectives and client state

`--akd_teacher embeddings` adds MAE between a client projection and teacher
features, weighted by `--kd_beta 10`. Temperature does not enter this objective.
The projection is private client state, cached by client identity and reused
across rounds; only classifier weights are aggregated. Feature widths must match
`--embeddings_dims`.

`--akd_teacher global_logits` freezes the incoming round-global classifier and
uses `T² KL(softmax(student/T) || softmax(global/T))`, with `--temperature 2`.
This follows the orientation printed in Eq. 9. Both modes use aligned pretrained
features to estimate local label noise. Distillation activates at or after
`--distil_round` when estimated disagreement is at least `--noise_threshold .15`.
Late-participating clients estimate noise on their first eligible round.

## NA-FedAvg weighting

Set `--est_noise_round 30 --percentile 75` to configure the energy-estimation
stage. The server requires energy scores from every logical client before
computing weights. The implementation counts
`local_energy >= percentile(global_energy)` as noisy and applies
`min(0.1 * n, n - noisy)` to affected clients.

The article's prose describes the opposite side of the energy threshold and
also presents `(1 - noise_fraction) * n` weighting. The explicit threshold and
capped rule above define this implementation's convention. Zero total impact
at initialization raises an error; a later sampled subset with zero impact
retains the previous global model and records a skipped update.

## Execution and saved state

The Flower App uses local Ray workers; `run.py` runs clients sequentially with
stock Flower message serialization. Both save configuration, versions, losses,
accuracies, sample counts, and client metrics to `results.json`, plus evaluated
global weights to `final-parameters.npz`. Each invocation owns its output and
cache directories. Use a fresh output path for every run.
