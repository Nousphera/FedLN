# Paper data, teachers, and optimizer settings

The maintained code supports `cifar10`, `cifar100`, `cifar10n`, `cifar100n`,
`fashionmnist`, `pathmnist`, `eurosat`, and `spcm`. The first three sections below
provide executable preparation paths. These are maintained implementations;
the exact historical teacher extraction scripts and all experiment manifests
were not recovered. No dataset or pretrained model is redistributed here.

## Vision data and human annotations

CIFAR-10/100 and FashionMNIST download through TensorFlow/Keras when first used.
Their training rows keep Keras's canonical order. Optionally place an NPZ under
`--data_root` named `cifar10.npz`, `cifar100.npz`, or `fashionmnist.npz`, with
`train_images`, `train_labels`, `test_images`, `test_labels`. Images must be
uint8; labels are integer class IDs. This also allows small offline fixtures.

For PathMNIST, download the [official 28×28 RGB NPZ](https://zenodo.org/records/10519652/files/pathmnist.npz?download=1)
as `data/pathmnist.npz` (official MD5 `a8b06965200029087d5bd730944a56c1`). The
loader uses the official 89,996 training and 7,180 test images and nine-class
space. It does not merge the 10,004-image validation split into training.
See [MedMNIST's dataset metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
for the CC BY 4.0 terms. The original paper's exact split manifest was not recovered.

CIFAR-N uses human training labels and the clean original test labels. Obtain
`CIFAR-10_human.pt` or `CIFAR-100_human.pt` from the
[official annotation repository](https://github.com/UCSC-REAL/cifar-10-100n/tree/49df7d8a69e355470c77c1c2f2424916325a394b/data).
Convert in the separate CLIP/PyTorch environment described below:

```bash
.venv-clip/bin/python prepare_human_labels.py --dataset cifar10n \
  --input downloads/CIFAR-10_human.pt --output data/cifar10-human.npz
# CIFAR-100N: --dataset cifar100n --input downloads/CIFAR-100_human.pt
```

Conversion uses restricted `torch.load(weights_only=True)` with only the NumPy
array types used by the official files allowed, validates all 50,000 rows, and
records source/output hashes. Training requires `--human_labels` and an explicit
`--human_noise_key`: CIFAR-10N accepts `worse_label`, `aggre_label`, or
`random_label1/2/3`; CIFAR-100N accepts `noisy_label`. The paper does not identify
the exact CIFAR-10N annotation key, so none is selected automatically. The
annotation file's `clean_label` must exactly match the loaded training labels.
TFDS's separately ordered annotation files are **not** interchangeable with
Keras/Python CIFAR order. Synthetic noise arguments do not add noise to CIFAR-N.

```bash
python run.py --method fedavg --dataset_name cifar10n \
  --human_labels data/cifar10-human.npz --human_noise_key worse_label \
  --num_clients 30 --num_rounds 200 --participation_rate .8 \
  --batch_size 128 --seed 42 --output runs/cifar10n-worse-seed42
```

## CLIP ViT-B/32 image features

Use a separate Python 3.11 or 3.12 environment to avoid PyTorch/Transformers and
TensorFlow 2.15 dependency conflicts. Prepare images using the training environment:

```bash
python prepare_features.py --dataset cifar10 --data-root data \
  --output data/cifar10-clip-input.npz
python3.12 -m venv .venv-clip
.venv-clip/bin/python -m pip install -r requirements-features-clip.txt
.venv-clip/bin/python extract_features.py --teacher clip \
  --input data/cifar10-clip-input.npz --output features/cifar10.npy
python run.py --method nnc --dataset_name cifar10 \
  --embeddings_dir "$PWD/features" --knn_neighbors 100 \
  --num_clients 30 --num_rounds 200 --participation_rate .8 \
  --train_epochs 1 --batch_size 128 --seed 42 --output runs/cifar10-nnc
```

The extractor pins [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)
to revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, uses its official RGB
resize/crop/normalization processor, and saves the 512-dimensional projected
image embeddings without external L2 normalization. This choice is explicit:
the historical extraction normalization was not recovered. The first use
downloads the checkpoint; `--local-files-only` uses a previously populated cache.
Use the target dataset name for both preparation and output, including CIFAR-N.
EuroSAT preparation reads `data/eurosat/eurosat.npz`; training also needs
`TFDS_DATA_DIR="$PWD/data"`. Its historical array order remains an unresolved
input requirement; the maintained extractor verifies the chosen array's order.

## Speech Commands and TRILLsson 3

The recovered training loader expects waveform records; a maintained exporter
now creates them from the published twelve-label TFDS recipe:

```bash
python -m pip install -r requirements-data-audio.txt
python prepare_audio.py --tfds-cache data/tfds --output data/spcm-prepared \
  --train-shards 100 --download
export TFDS_DATA_DIR="$PWD/data/spcm-prepared"
python3.11 -m venv .venv-trillsson
.venv-trillsson/bin/python -m pip install -r requirements-features-audio.txt
.venv-trillsson/bin/python extract_features.py --teacher trillsson3 \
  --tfds-data-dir "$TFDS_DATA_DIR" --output features/spcm.npy
python run.py --method akd --dataset_name spcm --model_name cnn \
  --embeddings_dir "$PWD/features" --embeddings_dims 1024 --knn_neighbors 100 \
  --num_clients 30 --num_rounds 200 --train_epochs 1 --batch_size 128 \
  --seed 42 --output runs/spcm-akd
```

Without `--download`, the exporter requires an existing TFDS cache. The full
corpus download is substantial. `--limit-per-split N` explicitly creates a
bounded fixture, not a benchmark dataset. The exporter uses
[TFDS 4.9.4 Speech Commands 0.0.3](https://github.com/tensorflow/datasets/blob/v4.9.4/tensorflow_datasets/datasets/speech_commands/speech_commands_dataset_builder.py),
which provides the ten commands plus `_silence_` and `_unknown_`; it uses its
published train/validation/test splits and background-noise recipe. Validation
is exported but is not consumed by the existing training runner. Training rows
are assigned dense deterministic IDs and round-robin record shards; source
TFDS IDs are preserved in sidecars. TFDS does not expose speaker IDs in this
schema; empty speaker fields are recorded explicitly. This is a maintained
recipe, not a recovered exact historical partition or the paper's controlled
25% quantity variance.

The [official TRILLsson 3 SavedModel](https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1)
uses EfficientNetV2-B3 and returns 1,024-dimensional embeddings. The extractor
passes complete mono 16 kHz PCM waveforms scaled by 32768, with no external L2
normalization; the SavedModel owns its frontend and context padding. This is
separate from the client's recovered L2-normalized random-window spectrogram
pipeline. The model artifact hash and preprocessing are recorded. `--checkpoint`
accepts a local SavedModel, whose actual hash is still recorded; callers are
responsible for supplying the intended teacher. See [the recovered record
schema](RECOVERED_DATA.md).

## Identity checks and saved evidence

Each extractor writes `<dataset>.npy`, `<dataset>.ids.npy`, and
`<dataset>.manifest.json`. Output reuse is rejected. Features are reordered by
stable source row IDs, and must cover each canonical training row exactly once.
The manifest records source-content/order and feature-content SHA-256 digests,
teacher identity/revision or artifact hash, preprocessing, and runtime versions.
NNC/AKD validate these artifacts against the current data before selecting
client rows. Audio loading additionally checks actual record IDs and labels
against the sidecars in record order. Altered order/content raises an error.

The two legacy bundled vision arrays lack these manifests and emit an explicit
unverified-alignment warning. Use newly extracted features for an auditable
run. Results retain configuration, runtime versions, input/feature hashes and
feature metadata, metrics, and final global parameters. A final parameter NPZ
is not a resumable optimizer/client-cache checkpoint.

## Optimizers and AKD objectives

| Dataset | `--optimizer auto` | Default learning rate |
| --- | --- | --- |
| CIFAR-10, FashionMNIST | SGD, momentum 0.9 | 0.1 |
| PathMNIST, EuroSAT, Speech Commands | Adam | 0.001 |
| CIFAR-10N/100/100N | SGD, momentum 0.9 | 0.1 |

The first two rows follow final-paper §5.2. CIFAR-N/100 reuse the CIFAR family
setting explicitly; the paper does not specify a separate optimizer for each
human-noise experiment. `--optimizer`, `--lr`, and `--momentum` override defaults
and are saved in results. The setting reaches all maintained client training
and confident-learning cross-validation. This corrects the old source's
unconditional Adam, so CIFAR trajectories change.

AKD defaults to `--akd_teacher embeddings`: the existing MAE between the local
projection and teacher features, weighted by `--kd_beta 10`. Temperature does
not affect MAE. `--akd_teacher global_logits` implements Eq. 9 using a frozen
copy of the **incoming round-global classifier**, with
`T² KL(softmax(student/T) || softmax(global/T))`, `--temperature 2`, and the same
coefficient. Its orientation follows the printed equation explicitly. The
classifier alone is aggregated; the teacher is neither updated nor uploaded.
Both modes still need aligned pretrained features to estimate each client's
noise once from its local labels. Distillation activates at/after
`--distil_round` when estimated disagreement reaches `--noise_threshold .15`.
The threshold comparison preserves the recovered `>=` convention. Global
teacher execution is a maintained implementation of the equation, not recovered
original experimental code.
