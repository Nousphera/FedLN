# FedLN

**Labeling Chaos to Learning Harmony: Federated Learning with Noisy Labels**
[Paper](https://doi.org/10.1145/3626242) · [Project page](https://nousphera.github.io/FedLN/) · [Code](https://github.com/Nousphera/FedLN) · [Source provenance](PROVENANCE.md)

## Abstract

Federated Learning (FL) is a distributed machine learning paradigm that enables learning models from decentralized private datasets where the labeling effort is entrusted to the clients. While most existing FL approaches assume high-quality labels are readily available on users’ devices, in reality, label noise can naturally occur in FL and is closely related to clients’ characteristics. Due to scarcity of available data and significant label noise variations among clients in FL, existing state-of-the-art centralized approaches exhibit unsatisfactory performance, whereas prior FL studies rely on excessive on-device computational schemes or additional clean data available on the server. We propose FedLN, a framework to deal with label noise across different FL training stages, namely FL initialization, on-device model training, and server model aggregation, able to accommodate the diverse computational capabilities of devices in an FL system. Specifically, FedLN computes per-client noise level estimation in a single federated round and improves the models’ performance by either correcting or mitigating the effect of noisy samples. Our evaluation on various publicly available vision and audio datasets demonstrates a 22% improvement on average compared to other existing methods for a label noise level of 60%. We further validate the efficiency of FedLN in human-annotated real-world noisy datasets and report a 4.8% increase on average in models’ recognition performance, highlighting that FedLN can be useful for improving FL services provided to everyday users.

*The abstract describes the published work. The maintained release has not reproduced its benchmark tables.*

## Setup

Use Python 3.11. The commands below create a dedicated virtual environment.
Flower 1.13.1 and Ray 2.10.0 are pinned; TensorFlow projects require the
compatible protobuf dependency range. Keep the checkout and virtual environment
on paths without spaces for Ray subprocesses. Do not upgrade Flower independently.

```bash
mkdir -p "$HOME/research-code" "$HOME/.venvs"
cd "$HOME/research-code"
git clone https://github.com/Nousphera/FedLN.git
cd FedLN
python3.11 -m venv "$HOME/.venvs/fedln"
source "$HOME/.venvs/fedln/bin/activate"
python -m pip install -r requirements.txt -r requirements-simulation.txt
python -m pytest -q
```

## Datasets

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

The same entry point covers CIFAR-100, FashionMNIST, PathMNIST, CIFAR-N,
EuroSAT and Speech Commands imports/validation. [Dataset setup](docs/DATA_SETUP.md)
lists every source and command. NNC/AKD additionally require aligned teacher
features: follow [CLIP/TRILLsson extraction](docs/PAPER_DATA.md), including the
separate teacher environment and explicit CIFAR-N annotation choice.

## Run

```bash
flwr run .
```

The default runs two rounds of NNC on a generated fixture with two clients.
This is an actual `ClientApp`/`ServerApp` simulation: it trains the original
clients, applies the original server strategy and writes `runs/flower-smoke`.
For prepared CIFAR-10 and newly extracted features:

```bash
flwr run . paper --run-config "options='--method nnc --dataset_name cifar10 --data_root data --embeddings_dir $PWD/features --num_clients 30 --num_rounds 200 --participation_rate .8 --train_epochs 1 --batch_size 128 --knn_neighbors 100 --output runs/cifar10-nnc'"
```

The same argument string accepts `--method akd`, `na_fedavg`, `fedavg`,
`bi_tempered_loss`, `label_smoothing` or `confidence_learning`.
See `python run.py --method METHOD --method-help`. For a sequential CPU check:

```bash
python run.py --method nnc --dataset_name synthetic --num_clients 2 \
  --num_rounds 2 --train_epochs 1 --batch_size 8 --knn_neighbors 5 --output runs/sequential
```

## Outputs and reproducibility

Every run requires a new output directory. `results.json` records configuration,
versions, sampled logical clients, example counts and round metrics;
`final-parameters.npz` saves the evaluated global weights. Native simulations
preserve run-owned caches under the output directory across worker lifetimes.
They are local simulations using a shared filesystem, not multi-host deployments.
They do not measure network latency. Final weights are not complete resumable
optimizer/client-state checkpoints. Synthetic metrics are execution checks only.

The `options` run-config value is the complete runner argument string. Set the selected federation's
`options.num-supernodes` to exactly the configured client count; mismatches fail
explicitly. [Flower setup details](docs/FLOWER_APP.md) document identity, state,
sampling and the tested runtime boundary.

The maintained methods cover label correction, embedding or global-logit AKD,
and noise-aware aggregation. The unresolved NA-FedAvg paper/source weighting
convention, teacher normalization and original partition gaps are documented in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [PAPER_DATA.md](docs/PAPER_DATA.md).
Teacher manifests bind row identity, source content and feature content. Legacy
bundled arrays have no verified alignment manifest.

## Citation

Use [CITATION.cff](CITATION.cff) or [citation.bib](citation.bib).

## License

[Apache-2.0](LICENSE). External datasets and pretrained models retain their own terms.
