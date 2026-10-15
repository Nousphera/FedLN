# Dataset preparation

Run commands from the repository root in the README's environment.
`python scripts/prepare_data.py --help` lists datasets and modes.
`--download` explicitly enables network acquisition; `--source` imports a safe
NPZ or already-prepared directory. Import rejects existing destination files and
symlinks in directory trees. Imports validate in a temporary staging directory before publishing; failed imports leave no dataset or manifest behind. Directory imports copy only the selected dataset subtree, and auxiliary imports copy only supported media files. `--validate` reads schema and content again and
compares the deterministic SHA-256 manifest; it never downloads. Omitting all
three modes registers an already-present dataset after validating its schema.

Prepared image files contain `train_images`, `train_labels`, `test_images`, and
`test_labels`. RGB images are uint8 NHWC with shape `(N,32,32,3)` for CIFAR. PathMNIST uses `(N,28,28,3)`; FashionMNIST uses uint8 `(N,28,28)`.
Labels are integer IDs shaped `(N,)` or `(N,1)`. Preparation checks shapes,
class ranges, nonempty splits, and counts. Training reads these files directly
and preserves their row order.

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --source /path/to/cifar10.npz
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

A manifest records dataset name, split sizes, class count and relative file
paths/content hashes. Retain the source publication, data license, and acquisition record alongside
this manifest. A changed file requires a new destination/preparation run;
manifests are not silently regenerated over changed data.

## Public vision sources

[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) use the official
train/test splits via the pinned Keras loader. Files are saved
as `ROOT/cifar10.npz` and `ROOT/cifar100.npz` for training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar100 --root datasets --download
```

PathMNIST uses the official 28x28 RGB NPZ from
[MedMNIST](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py),
[Zenodo record 10519652](https://zenodo.org/records/10519652).
Download checks MD5 `a8b06965200029087d5bd730944a56c1` before import. The
89,996-image train and 7,180-image test splits are used; the official validation
split is retained in the downloaded NPZ but is not merged into training.

```bash
python scripts/prepare_data.py --dataset pathmnist --root datasets --download
python scripts/prepare_data.py --dataset pathmnist --root datasets --validate
```
The trainer reads `datasets/pathmnist.npz` with `--data_root datasets`.

## FashionMNIST and human-noise annotations

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) downloads through
Keras into `ROOT/fashionmnist.npz`:

```bash
python scripts/prepare_data.py --dataset fashionmnist --root data --download
python scripts/prepare_data.py --dataset fashionmnist --root data --validate
```

For CIFAR-N, first prepare clean CIFAR-10/100 and convert official annotations
with `prepare_human_labels.py` in the separate PyTorch environment documented in
[PAPER_DATA.md](PAPER_DATA.md). No annotation key is silently chosen:

```bash
python scripts/prepare_data.py --dataset cifar10n --root data \
  --human-labels data/cifar10-human.npz --human-noise-key worse_label
python scripts/prepare_data.py --dataset cifar10n --root data --validate \
  --human-labels data/cifar10-human.npz --human-noise-key worse_label
```

For CIFAR-100N use `cifar100n`, the converted CIFAR-100 annotation NPZ and
`--human-noise-key noisy_label`. Validation checks clean-label equality in
canonical row order and records the annotation hash/key. The trainer needs the
same `--human_labels` and `--human_noise_key` values. TFDS annotation order is
not interchangeable with canonical Keras/Python CIFAR order.

## EuroSAT

Obtain RGB imagery from the [official EuroSAT project](https://github.com/phelber/EuroSAT).
Import a prepared directory containing `eurosat/eurosat.npz` with `images`
(27000,64,64,3) uint8 and `labels` (27000,) or (27000,1) integer IDs 0-9:

```bash
python scripts/prepare_data.py --dataset eurosat --source /path/to/prepared-root --root data/eurosat-prepared
python scripts/prepare_data.py --dataset eurosat --root data/eurosat-prepared --validate
export TFDS_DATA_DIR="$PWD/data/eurosat-prepared"
```

The loader takes the first 21,600 rows for training and the remaining 5,400
for testing, preserving the supplied array order. The class space is fixed to
ten. Extract CLIP features from this same file using [PAPER_DATA.md](PAPER_DATA.md)
so source hashes and row IDs match; features from a differently ordered array
cannot be substituted.

## Speech Commands and teacher features

```bash
python -m pip install -r requirements-data-audio.txt
python prepare_audio.py --tfds-cache data/tfds --output data/spcm-prepared --train-shards 100 --download
python scripts/prepare_data.py --dataset spcm --root data/spcm-prepared
python scripts/prepare_data.py --dataset spcm --root data/spcm-prepared --validate
export TFDS_DATA_DIR="$PWD/data/spcm-prepared"
```

The exporter uses TFDS Speech Commands 0.0.3 and its published 12-label mapping.
Without `--download`, it requires an existing TFDS cache. The validator checks
train/test record IDs and labels against identity sidecars, then hashes all
prepared files. Importing with `--source /path/to/prepared-root` retains the same relative paths and content hashes after relocation; validation restores any pre-existing `TFDS_DATA_DIR` setting. Set it explicitly to the final root for training. Training clients cannot exceed the number of train shards.
`--limit-per-split` creates bounded fixtures for testing. Run CLIP or TRILLsson extraction in the separate environments described
in [PAPER_DATA.md](PAPER_DATA.md); keep teacher and source manifests together.
