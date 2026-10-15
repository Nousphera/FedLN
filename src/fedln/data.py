"""Canonical, row-aligned datasets for training and teacher extraction."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import wave

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

DATASETS = ("cifar10", "cifar100", "cifar10n", "cifar100n", "fashionmnist", "pathmnist", "eurosat", "spcm")
CLASSES = dict(zip(DATASETS, (10, 100, 10, 100, 10, 9, 10, 12)))
SPEECH_LABELS = ("down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "_silence_", "_unknown_")
HUMAN_KEYS = {"cifar10n": ("aggre_label", "worse_label", "random_label1", "random_label2", "random_label3"),
              "cifar100n": ("noisy_label",)}


def checked_path(root, name, parent_cache=None):
    """Resolve only regular local data files; reject traversal and symbolic links."""
    relative = PurePosixPath(str(name))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in str(name):
        raise ValueError(f"Unsafe data path: {name}")
    cache = {} if parent_cache is None else parent_cache
    root = Path(root)
    root_key = ("root", root)
    if root_key not in cache:
        cache[root_key] = root.resolve(strict=True)
    root = cache[root_key]
    path = root / Path(*relative.parts)
    if path.parent not in cache:
        resolved = path.parent.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved != path.parent:
            raise ValueError(f"Data paths cannot traverse symbolic links: {name}")
        cache[path.parent] = resolved
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Data file is missing or is a symbolic link: {name}")
    return path


def _check_media_paths(root, paths):
    cache = {}
    for path in np.unique(paths):
        checked_path(root, str(path), cache)


def read_waveform(path, start=0, length=-1):
    """Read the corpus's original signed 16-bit, mono 16 kHz PCM."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"Expected mono 16 kHz, 16-bit PCM: {path}")
        if start < 0 or start >= stream.getnframes():
            raise ValueError(f"Invalid waveform offset: {path}")
        count = stream.getnframes() - start if length < 0 else length
        if count < 1 or start + count > stream.getnframes():
            raise ValueError(f"Invalid waveform length: {path}")
        stream.setpos(start)
        samples = stream.readframes(count)
    if len(samples) != 2 * count:
        raise ValueError(f"Truncated waveform: {path}")
    return torch.from_numpy(np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0)


def image_tensor(image, dataset, training):
    """Preserve the released image preprocessing, including origin-centered Cutout."""
    x = TF.pil_to_tensor(image).float().div_(255)
    if training and dataset in ("cifar10", "cifar100", "cifar10n", "cifar100n", "eurosat", "fashionmnist"):
        if bool(torch.rand(()) < 0.5):
            x = x.flip(-1)
        if dataset != "fashionmnist":
            size = x.shape[-1]
            x = TF.pad(x, [4, 4, 4, 4])
            top, left = torch.randint(9, (2,)).tolist()
            x = x[:, top:top + size, left:left + size]
    if dataset.startswith("cifar"):
        x = TF.normalize(x, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == "pathmnist":
        x = TF.normalize(x, (0.5,) * 3, (0.5,) * 3)
    if training:
        extent = 16 if dataset == "eurosat" else 8 if dataset.startswith("cifar") else 7
        x[:, :extent, :extent] = 0
    return x


class VisionDataset(Dataset):
    def __init__(self, directory, root, name, split, labels, training):
        self.root, self.name, self.training = Path(root).resolve(), name, training
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ids = np.arange(len(self.labels), dtype=np.int64)
        image_path = directory / f"{split}_images.npy"
        self.images = np.load(image_path, mmap_mode="r", allow_pickle=False) if image_path.exists() else None
        self.paths = np.load(directory / f"{split}_paths.npy", allow_pickle=False) if self.images is None else None
        if self.paths is not None:
            _check_media_paths(self.root, self.paths)
        if len(self.images if self.images is not None else self.paths) != len(self.labels):
            raise ValueError("Images and labels have different row counts")

    def __len__(self):
        return len(self.labels)

    def raw(self, index):
        if self.images is not None:
            return Image.fromarray(np.asarray(self.images[index]))
        with Image.open(self.root / str(self.paths[index])) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        return image_tensor(self.raw(index), self.name, self.training), int(self.labels[index]), int(index)


class AudioDataset(Dataset):
    def __init__(self, directory, root, split, labels, training):
        self.root, self.training = Path(root).resolve(), training
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ids = np.arange(len(self.labels), dtype=np.int64)
        self.paths = np.load(directory / f"{split}_paths.npy", allow_pickle=False)
        _check_media_paths(self.root, self.paths)
        self.offsets = np.load(directory / f"{split}_offsets.npy", mmap_mode="r", allow_pickle=False)
        if len(self.paths) != len(self.labels) or self.offsets.shape != (len(self.labels), 2):
            raise ValueError("Waveform rows, offsets and labels do not align")

    def __len__(self):
        return len(self.labels)

    def raw(self, index):
        start, count = self.offsets[index]
        return read_waveform(self.root / str(self.paths[index]), int(start), int(count))

    def __getitem__(self, index):
        from .audio import audio_spectrogram
        return audio_spectrogram(self.raw(index), training=self.training), int(self.labels[index]), int(index)


@dataclass(frozen=True)
class DataBundle:
    train: Dataset
    test: Dataset
    labels: np.ndarray
    classes: int
    input_channels: int
    fingerprint: str
    modality: str


def load_data(dataset, root="data", training=True, human_noise_key=None):
    """Load prepared data without downloads; row IDs index the canonical train split."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")
    root = Path(root).expanduser().resolve()
    directory = root / "prepared" / dataset
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare {dataset} first: python scripts/prepare_data.py --dataset {dataset} --root {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or manifest.get("dataset") != dataset:
        raise ValueError(f"Unsupported preparation manifest: {manifest_path}")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != manifest["fingerprint"]:
        raise ValueError(f"Preparation manifest was modified: {manifest_path}")
    for entry in manifest["files"]:
        path = checked_path(directory, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Prepared file missing or changed: {path}; rerun preparation")
    # Human variants reuse exactly the canonical clean image files.
    expected_base = {"cifar10n": "cifar10", "cifar100n": "cifar100"}.get(dataset, dataset)
    if manifest.get("image_dataset", dataset) != expected_base:
        raise ValueError("Unexpected image dataset in preparation manifest")
    image_directory = root / "prepared" / expected_base
    if dataset in HUMAN_KEYS:
        image_manifest = json.loads((image_directory / "manifest.json").read_text())
        if image_manifest["fingerprint"] != manifest["image_fingerprint"]:
            raise ValueError("Human annotation image fingerprint changed")
    clean = np.load(image_directory / "train_labels.npy", mmap_mode="r", allow_pickle=False)
    labels = clean
    if dataset in HUMAN_KEYS:
        key = human_noise_key or ("worse_label" if dataset == "cifar10n" else "noisy_label")
        if key not in HUMAN_KEYS[dataset]:
            raise ValueError(f"--human-noise-key must be one of {', '.join(HUMAN_KEYS[dataset])}")
        labels = np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
    elif human_noise_key is not None:
        raise ValueError("Human annotation keys apply only to CIFAR-10N/100N")
    test_labels = np.load(image_directory / "test_labels.npy", mmap_mode="r", allow_pickle=False)
    for values in (labels, test_labels):
        if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= CLASSES[dataset])):
            raise ValueError("Invalid prepared class labels")
    factory = AudioDataset if dataset == "spcm" else VisionDataset
    arguments = () if dataset == "spcm" else (dataset,)
    train = factory(image_directory, root, *arguments, "train", labels, training)
    test = factory(image_directory, root, *arguments, "test", test_labels, False)
    fingerprint = hashlib.sha256((manifest["fingerprint"] + hashlib.sha256(labels.tobytes()).hexdigest()).encode()).hexdigest()
    return DataBundle(train, test, labels, CLASSES[dataset], 1 if dataset in ("fashionmnist", "spcm") else 3,
                      fingerprint, "audio" if dataset == "spcm" else "vision")
