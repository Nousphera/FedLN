# FedLN: Federated Learning with Noisy Labels

FedLN estimates label noise on federated clients and uses those estimates during initialization, local training, or server aggregation. This repository contains the TensorFlow/Flower research implementation, with CIFAR-10 and EuroSAT loaders and CNN/ResNet20 models.

This independent Nousphera copy preserves the history and attribution of [FederatedML/FedLN](https://github.com/FederatedML/FedLN); see [provenance](PROVENANCE.md). Maintenance repairs seed propagation, label correction, preprocessing and transport errors while retaining the original history and archival dependency pins. These repairs can change numerical results.

## Paper

**Labeling Chaos to Learning Harmony: Federated Learning with Noisy Labels**

Vasileios Tsouvalas, Aaqib Saeed, Tanir Özçelebi, Nirvana Meratnia

*ACM Transactions on Intelligent Systems and Technology*, 15(2), Article 22, 2024.

[Journal article](https://doi.org/10.1145/3626242) · [Preprint](https://arxiv.org/abs/2208.09378) · [Original project page](https://federatedml.github.io/FedLN/) · [2022 workshop version](https://openreview.net/forum?id=gNHMC4I0Pva)

The journal evaluates a wider set of vision/audio datasets than the loaders shipped here. Paper results have not been reproduced during this maintenance work.

## Quick start

Python 3.11, TensorFlow 2.15.1 and stock Flower 1.13.1 are used by the maintained
CPU runner. Original clients train sequentially with standard Flower message
serialization, without a Ray cluster or patched framework.

```bash
git clone https://github.com/Nousphera/FedLN.git
cd FedLN
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --method nnc --dataset_name synthetic --num_clients 2 \
  --num_rounds 2 --train_epochs 1 --batch_size 8 --knn_neighbors 5 \
  --output runs/nnc-smoke
python -m pytest -q
```

Select `--method akd`, `na_fedavg` or `fedavg` for the other primary paths.
For the short NA-FedAvg check add `--est_noise_round 1`. Use
`--method-help` for all method arguments. The synthetic fixture tests execution;
it is not a pretrained embedding or a paper result.

[Reproduction notes](docs/REPRODUCIBILITY.md) provide real-data commands,
data alignment requirements, numerical fixes, and the unresolved NA-FedAvg
paper/source convention. The original Ray launcher remains at `src/run.sh`,
and its historical dependency list remains at `src/requirements.txt`.

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

Use `python run.py --method METHOD --method-help` for its arguments; centralized training remains an archival entry point. The registered dataset names are `cifar10` and `eurosat`; model names are `cnn` and `resnet20`.

## Data and files

- **CIFAR-10:** images download through the Keras dataset loader. Included `src/data/features/cifar10.npy` has 50,000 rows and 512 float16 features.
- **EuroSAT:** supply `src/data/eurosat/eurosat.npz`, or set `TFDS_DATA_DIR` to a directory containing `eurosat/eurosat.npz`. The archive must contain `images` and `labels` in the original order. Included embeddings have 27,000 rows and 512 float16 features; the loader uses the first 21,600 examples for training. The archive and an independently verified preparation recipe are not included.
- **Embeddings:** NNC/AKD use `--embeddings_dir` (default `./data/features`). Extraction provenance and row alignment must be checked before substituting new data/features.
- **Temporary state:** each cached method creates a private `fedln-*` child of `--temp_dir`. Normal completion and Python exceptions clean up that child; existing files in the parent are preserved. These caches are not checkpoints.

`src/fedln/` contains the three methods; `src/baselines/` contains comparisons; `src/data/`, `src/models/`, and `src/utils/` hold the shared implementation. `index.html` and `img/` contain the project page and original research figures.

## Reproducibility and checks

The tests include real CPU training for FedAvg, NNC, AKD and NA-FedAvg,
standard Flower serialization, numerical fixtures and the original launcher
regressions. See [reproduction notes](docs/REPRODUCIBILITY.md) and
[validation](docs/VALIDATION.md) for scope. Published accuracy tables, GPU
execution, multi-host scheduling and the full journal dataset suite have not
been reproduced.

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
