# FedLN: Federated Learning with Noisy Labels

FedLN estimates label noise on federated clients and uses those estimates during initialization, local training, or server aggregation. This repository contains the TensorFlow/Flower research implementation, with CIFAR-10 and EuroSAT loaders and CNN/ResNet20 models.

This independent Nousphera copy preserves the history and attribution of [FederatedML/FedLN](https://github.com/FederatedML/FedLN); see [provenance](PROVENANCE.md). Maintenance changes address launch/setup failures and documentation; the research algorithms and historical dependency pins remain unchanged.

## Paper

**Labeling Chaos to Learning Harmony: Federated Learning with Noisy Labels**

Vasileios Tsouvalas, Aaqib Saeed, Tanir Özçelebi, Nirvana Meratnia

*ACM Transactions on Intelligent Systems and Technology*, 15(2), Article 22, 2024.

[Journal article](https://doi.org/10.1145/3626242) · [Preprint](https://arxiv.org/abs/2208.09378) · [Original project page](https://federatedml.github.io/FedLN/) · [2022 workshop version](https://openreview.net/forum?id=gNHMC4I0Pva)

The journal evaluates a wider set of vision/audio datasets than the loaders shipped here. Paper results have not been reproduced during this maintenance work.

## Quick start

```bash
git clone https://github.com/Nousphera/FedLN.git
cd FedLN
./src/run.sh --help
./src/run.sh nnc --help
```

For training, use an isolated legacy environment on an NVIDIA-equipped machine. Python 3.10 is a candidate based on published TensorFlow/Ray wheels; a fresh installation and training run have **not** been validated. The retained pins include a Flower/Ray simulation-version discrepancy; read [reproduction notes](docs/REPRODUCIBILITY.md) before installing.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
python -m pip check
```

The historical CIFAR-10 experiment invocation is:

```bash
./src/run.sh nnc --dataset_name cifar10 --model_name resnet20 \
  --num_rounds 100 --num_clients 10 --noisy_frac 0.8 \
  --noise_level 0.4 --noise_sparsity 0.7
```

Replace `nnc` with another method below. These are experiment entry points, not verified benchmark recipes. The launcher resolves relative data, embedding, and cache paths from `src/`; absolute paths avoid ambiguity. It also works as `./run.sh` from `src/`.

## Methods

| Command | Approach | Extra input |
| --- | --- | --- |
| `nnc` | Nearest Neighbor Correction | Aligned pretrained embeddings |
| `akd` | Adaptive Knowledge Distillation | Aligned pretrained embeddings |
| `na_fedavg` | Noise-Aware Federated Averaging | No pretrained embeddings |
| `fedavg` | Federated Averaging | — |
| `bi_tempered_loss` | FedAvg with bi-tempered logistic loss | — |
| `label_smoothing` | FedAvg with label smoothing | — |
| `confidence_learning` | FedAvg with confident learning | Local cross-validation |
| `centralized` | Centralized training | Uses `--epochs`, not `--num_rounds` |

Use `METHOD --help` for its arguments. The registered dataset names are `cifar10` and `eurosat`; model names are `cnn` and `resnet20`.

## Data and files

- **CIFAR-10:** images download through the Keras dataset loader. Included `src/data/features/cifar10.npy` has 50,000 rows and 512 float16 features.
- **EuroSAT:** supply `src/data/eurosat/eurosat.npz`, or set `TFDS_DATA_DIR` to a directory containing `eurosat/eurosat.npz`. The archive must contain `images` and `labels` in the original order. Included embeddings have 27,000 rows and 512 float16 features; the loader uses the first 21,600 examples for training. The archive and an independently verified preparation recipe are not included.
- **Embeddings:** NNC/AKD use `--embeddings_dir` (default `./data/features`). Extraction provenance and row alignment must be checked before substituting new data/features.
- **Temporary state:** each cached method creates a private `fedln-*` child of `--temp_dir`. Normal completion and Python exceptions clean up that child; existing files in the parent are preserved. These caches are not checkpoints.

`src/fedln/` contains the three methods; `src/baselines/` contains comparisons; `src/data/`, `src/models/`, and `src/utils/` hold the shared implementation. `index.html` and `img/` contain the project page and original research figures.

## Reproducibility and checks

[Reproduction notes](docs/REPRODUCIBILITY.md) record the legacy environment, data requirements, and known seed/KNN/distillation behaviors retained for scientific review. CPU/Apple GPU execution and complete distributed training remain unvalidated.

```bash
python -m unittest discover -s tests -v
```

Runtime regressions use the standard library and mocked training/GPU calls. Three data-utility checks use NumPy and are skipped if it is unavailable. Passing these checks does not establish model accuracy or paper reproduction.

## Citation

See [CITATION.cff](CITATION.cff). The final journal publication is from 2024; the accepted manuscript appeared online in 2023.

```bibtex
@article{tsouvalas2024fedln,
  author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir and Meratnia, Nirvana},
  title = {Labeling Chaos to Learning Harmony: Federated Learning with Noisy Labels},
  journal = {ACM Transactions on Intelligent Systems and Technology},
  year = {2024},
  volume = {15},
  number = {2},
  articleno = {22},
  numpages = {26},
  doi = {10.1145/3626242}
}
```

## License

The source retains its original [Apache License 2.0](LICENSE). Cite the authors when using this research implementation; external datasets retain their own terms.
<!-- nousphera-project-resources -->
## Project resources

[Project page](https://nousphera.github.io/FedLN/) · [Research overview](https://nousphera.github.io/projects/fedln/) · [A0 poster](https://nousphera.github.io/FedLN/poster.pdf) · [Paper](https://doi.org/10.1145/3626242)

This repository contains the research implementation. The project page and poster summarize the published method and reported experiments; this documentation update does not claim an independent reproduction of those experiments.
