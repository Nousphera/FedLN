"""Download, verify and prepare the paper datasets without TensorFlow."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import wave
import zipfile

import numpy as np

from .data import CLASSES, DATASETS, HUMAN_KEYS, SPEECH_LABELS, checked_path

FASHION = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
SPEECH = {
    "speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
    "speech_commands_test_set_v0.02.tar.gz": "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df",
}
HUMAN = {
    "cifar10n": ("CIFAR-10_human.pt", "873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310"),
    "cifar100n": ("CIFAR-100_human.pt", "bd2d80409754d420292d622e15e25248ba21e37d27429efac49c8da723f44394"),
}


def digest(path, algorithm="sha256"):
    checksum = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(url, destination, checksum, algorithm="sha256"):
    """Resume partial downloads; publish only after the official checksum matches."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or digest(destination, algorithm) != checksum:
            raise ValueError(f"Checksum mismatch: {destination}; remove this damaged download and retry")
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"Download destination is a symbolic link: {partial}")
    if partial.exists() and digest(partial, algorithm) == checksum:
        partial.replace(destination)
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Nousphera-FedLN/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        resume = offset > 0 and response.status == 206
        if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server returned an incorrect download range")
        with partial.open("ab" if resume else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if digest(partial, algorithm) != checksum:
        raise ValueError(f"Checksum mismatch: {partial}; remove this partial file and retry")
    partial.replace(destination)
    return destination


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or (not path.parts and name not in (".", "./")):
        raise ValueError(f"Unsafe archive member: {name}")
    return Path(*path.parts)


def extract_archive(archive, destination):
    """Extract verified archives atomically, rejecting links and special files."""
    archive, destination = Path(archive), Path(destination)
    checksum = digest(archive)
    marker = destination / ".archive.json"
    if marker.is_file():
        if json.loads(marker.read_text()).get("sha256") != checksum:
            raise ValueError(f"Extraction belongs to another archive: {destination}")
        return destination
    if destination.exists():
        raise FileExistsError(f"Unverified extraction exists: {destination}; choose an empty data root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for info in source.infolist():
                    target = stage / safe_member(info.filename)
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode not in (0, 0o040000, 0o100000):
                        raise ValueError(f"Archive links and special files are not supported: {info.filename}")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source.open(info) as reader, target.open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
        else:
            with tarfile.open(archive) as source:
                for member in source:
                    target = stage / safe_member(member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(member) as reader, target.open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
                    else:
                        raise ValueError(f"Archive links and special files are not supported: {member.name}")
        (stage / ".archive.json").write_text(json.dumps({"sha256": checksum}) + "\n")
        stage.rename(destination)
    return destination


def save_array(directory, name, values):
    with (Path(directory) / f"{name}.npy").open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)


def _file_entry(path, root):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Data files cannot be symbolic links: {path}")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}


def finish_manifest(directory, dataset, metadata, raw_files=(), root=None):
    files = [_file_entry(path, directory) for path in sorted(Path(directory).glob("*.npy"))]
    external = [_file_entry(path, root) for path in sorted(set(raw_files))]
    manifest = {"schema": 1, "dataset": dataset, "classes": CLASSES[dataset],
                "files": files, "raw_files": external, **metadata}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (Path(directory) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_prepared(root, dataset):
    root = Path(root).expanduser().resolve()
    directory = root / "prepared" / dataset
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest.pop("fingerprint")
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != claimed:
        raise ValueError(f"Preparation manifest was modified: {manifest_path}")
    if manifest["dataset"] != dataset or manifest["schema"] != 1:
        raise ValueError(f"Unexpected preparation manifest: {manifest_path}")
    for base, entries in ((directory, manifest["files"]), (Path(root), manifest["raw_files"])):
        parent_cache = {}
        for entry in entries:
            path = checked_path(base, entry["path"], parent_cache)
            if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Prepared data is missing or changed: {path}")
    manifest["fingerprint"] = claimed
    return manifest


def verify_dataset(dataset, root="data"):
    """Hash-check data once before launching workers or extracting embeddings."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(root).expanduser().resolve()
    manifest = verify_prepared(root, dataset)
    if dataset in HUMAN_KEYS:
        expected = "cifar10" if dataset == "cifar10n" else "cifar100"
        if manifest.get("image_dataset") != expected:
            raise ValueError("Unexpected image dataset for human annotations")
        base = verify_prepared(root, expected)
        if base["fingerprint"] != manifest["image_fingerprint"]:
            raise ValueError("Human annotation image fingerprint changed")
    return manifest


@contextmanager
def staging(root, dataset):
    destination = Path(root) / "prepared" / dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Partial preparation exists: {destination}; remove it before retrying")
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=destination.parent) as temporary:
        stage = Path(temporary) / dataset
        stage.mkdir()
        yield stage
        stage.rename(destination)


def _prepare_cifar(root, dataset, directory):
    from torchvision.datasets import CIFAR10, CIFAR100
    cls = CIFAR10 if dataset == "cifar10" else CIFAR100
    archive = download(cls.url, root / "downloads" / cls.filename, cls.tgz_md5, "md5")
    raw = extract_archive(archive, root / "raw" / dataset)
    counts = {}
    for split in ("train", "test"):
        data = cls(raw, train=split == "train", download=False)
        save_array(directory, f"{split}_images", data.data)
        save_array(directory, f"{split}_labels", np.asarray(data.targets, dtype=np.int64))
        counts[split] = len(data)
    return finish_manifest(directory, dataset, {"rows": counts, "source": cls.url, "source_sha256": digest(archive),
                                               "row_order": "Official Python batch order"})


def _read_idx(path):
    with gzip.open(path, "rb") as stream:
        header = stream.read(4)
        if len(header) != 4 or header[:3] != b"\x00\x00\x08" or header[3] not in (1, 3):
            raise ValueError(f"Invalid unsigned-byte IDX header: {path}")
        shape = struct.unpack(">" + "I" * header[3], stream.read(4 * header[3]))
        if np.prod(shape) > 100_000_000:
            raise ValueError("Unexpected IDX dimensions")
        payload = stream.read()
    if len(payload) != int(np.prod(shape)):
        raise ValueError(f"Truncated IDX data: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


def _prepare_fashion(root, directory):
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    artifacts = {name: download(base + name, root / "downloads" / name, checksum) for name, checksum in FASHION.items()}
    for split, prefix, expected in (("train", "train", 60000), ("test", "t10k", 10000)):
        images = _read_idx(artifacts[f"{prefix}-images-idx3-ubyte.gz"])
        labels = _read_idx(artifacts[f"{prefix}-labels-idx1-ubyte.gz"]).astype(np.int64)
        if images.shape != (expected, 28, 28) or labels.shape != (expected,) or np.any(labels > 9):
            raise ValueError("Unexpected Fashion-MNIST dimensions or labels")
        save_array(directory, f"{split}_images", images)
        save_array(directory, f"{split}_labels", labels)
    return finish_manifest(directory, "fashionmnist", {"rows": {"train": 60000, "test": 10000},
                           "source": base, "source_sha256": FASHION, "row_order": "Official IDX order"})


def _prepare_pathmnist(root, directory):
    url = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
    source = download(url, root / "downloads" / "pathmnist.npz", "a8b06965200029087d5bd730944a56c1", "md5")
    counts = {}
    with np.load(source, allow_pickle=False) as arrays:
        for split, expected in (("train", 89996), ("val", 10004), ("test", 7180)):
            images, labels = arrays[f"{split}_images"], arrays[f"{split}_labels"].reshape(-1)
            if images.shape != (expected, 28, 28, 3) or images.dtype != np.uint8 or len(labels) != expected or np.any((labels < 0) | (labels > 8)):
                raise ValueError("Unexpected official PathMNIST images or labels")
            name = "validation" if split == "val" else split
            save_array(directory, f"{name}_images", images)
            save_array(directory, f"{name}_labels", labels.astype(np.int64))
            counts[name] = expected
    return finish_manifest(directory, "pathmnist", {"rows": counts, "source": url, "source_sha256": digest(source),
                                                  "row_order": "Official MedMNIST 28x28 split order"})


def _prepare_human(root, dataset, directory):
    import torch
    base = "cifar10" if dataset == "cifar10n" else "cifar100"
    prepare_dataset(base, root)
    filename, checksum = HUMAN[dataset]
    url = f"https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/{filename}"
    source = download(url, root / "downloads" / filename, checksum)
    allowed = [(np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               (np.core.multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
               np.ndarray, np.dtype, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        arrays = torch.load(source, map_location="cpu", weights_only=True)
    clean = np.load(root / "prepared" / base / "train_labels.npy", allow_pickle=False)
    if not np.array_equal(np.asarray(arrays["clean_label"]), clean):
        raise ValueError("Human annotation order differs from official CIFAR Python order")
    for key in HUMAN_KEYS[dataset]:
        labels = np.asarray(arrays[key])
        if labels.shape != (50000,) or labels.dtype != np.int64 or np.any((labels < 0) | (labels >= CLASSES[dataset])):
            raise ValueError(f"Invalid human annotation array: {key}")
        save_array(directory, key, labels)
    base_manifest = verify_prepared(root, base)
    return finish_manifest(directory, dataset, {"image_dataset": base, "image_fingerprint": base_manifest["fingerprint"],
                           "rows": base_manifest["rows"], "source": url, "source_sha256": checksum,
                           "annotation_keys": HUMAN_KEYS[dataset], "row_order": "Official CIFAR Python order"})


def _prepare_eurosat(root, directory):
    from PIL import Image
    url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip"
    archive = download(url, root / "downloads" / "EuroSAT.zip", "c8fa014336c82ac7804f0398fcb19387", "md5")
    raw = extract_archive(archive, root / "raw" / "eurosat")
    classes = sorted(path for path in (raw / "2750").iterdir() if path.is_dir())
    if len(classes) != 10:
        raise ValueError("Expected ten EuroSAT RGB classes")
    paths, labels = [], []
    for label, folder in enumerate(classes):
        for path in sorted(folder.glob("*.jpg")):
            with Image.open(path) as image:
                if image.size != (64, 64) or image.mode != "RGB":
                    raise ValueError(f"Invalid EuroSAT RGB image: {path}")
                image.verify()
            paths.append(path)
            labels.append(label)
    if len(paths) != 27000:
        raise ValueError("Expected 27,000 EuroSAT images")
    order = np.random.default_rng(42).permutation(len(paths))
    relative = np.asarray([str(path.relative_to(root)) for path in paths])
    labels = np.asarray(labels, dtype=np.int64)
    for split, indices in (("train", order[:21600]), ("test", order[21600:])):
        save_array(directory, f"{split}_paths", relative[indices])
        save_array(directory, f"{split}_labels", labels[indices])
    return finish_manifest(directory, "eurosat", {"rows": {"train": 21600, "test": 5400}, "source": url,
                           "source_sha256": digest(archive), "class_names": [path.name for path in classes],
                           "split": "80/20 random split; NumPy PCG64 seed 42; sorted source paths"}, paths, root)


def speech_rows(main, test):
    """Follow Speech Commands 0.0.3's 12-class split and background policy."""
    main, test = Path(main), Path(test)
    validation = set((main / "validation_list.txt").read_text().splitlines())
    excluded = set((main / "testing_list.txt").read_text().splitlines())
    validation.add("_background_noise_/running_tap.wav")
    rows = {"train": [], "validation": [], "test": []}
    for source, test_set in ((main, False), (test, True)):
        for path in sorted(source.rglob("*.wav")):
            relative = path.relative_to(source).as_posix()
            word = path.parent.name
            if not test_set and relative in excluded:
                continue
            split = "test" if test_set else "validation" if relative in validation else "train"
            label = SPEECH_LABELS.index(word) if word in SPEECH_LABELS else 10 if word == "_background_noise_" else 11
            with wave.open(str(path), "rb") as stream:
                if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
                    raise ValueError(f"Invalid Speech Commands PCM: {path}")
                count = stream.getnframes()
            if not count:
                raise ValueError(f"Empty waveform: {path}")
            if word == "_background_noise_":
                for offset in range(0, count - 16000, 8000):
                    rows[split].append((path, label, offset, 16000))
            else:
                rows[split].append((path, label, 0, count))
    speakers = {split: {row[0].stem.split("_nohash_")[0] for row in values if "_nohash_" in row[0].stem}
                for split, values in rows.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if speakers[left] & speakers[right]:
            raise ValueError(f"Speech Commands speaker overlap between {left} and {right}")
    if any(not values for values in rows.values()):
        raise ValueError("Speech Commands requires nonempty train, validation and test splits")
    return rows


def _prepare_speech(root, directory):
    artifacts = {name: download("https://storage.googleapis.com/download.tensorflow.org/data/" + name,
                               root / "downloads" / name, checksum) for name, checksum in SPEECH.items()}
    main = extract_archive(artifacts["speech_commands_v0.02.tar.gz"], root / "raw" / "speech_commands_v0.02")
    test = extract_archive(artifacts["speech_commands_test_set_v0.02.tar.gz"], root / "raw" / "speech_commands_test_set_v0.02")
    rows = speech_rows(main, test)
    files = []
    for split, values in rows.items():
        save_array(directory, f"{split}_paths", [str(row[0].relative_to(root)) for row in values])
        save_array(directory, f"{split}_labels", np.asarray([row[1] for row in values], dtype=np.int64))
        save_array(directory, f"{split}_offsets", np.asarray([row[2:] for row in values], dtype=np.int64))
        files.extend(row[0] for row in values)
    return finish_manifest(directory, "spcm", {"rows": {split: len(values) for split, values in rows.items()},
                           "source_sha256": SPEECH, "class_names": SPEECH_LABELS, "sample_rate": 16000,
                           "split": "Official validation/testing lists; separate official test archive; speaker-disjoint",
                           "unknown": "All non-target words", "silence": "1-second background windows, 0.5-second stride; running_tap reserved for validation",
                           "row_order": "Sorted source paths, then increasing waveform offset"}, files, root)


def prepare_dataset(dataset, root="data", validate=False):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(root).expanduser().resolve()
    if (root / "prepared" / dataset / "manifest.json").is_file():
        manifest = verify_dataset(dataset, root)
        print(f"{dataset}: ready ({manifest['rows']})", flush=True)
        return manifest
    if validate:
        raise FileNotFoundError(f"{dataset} is not prepared under {root}")
    with staging(root, dataset) as directory:
        if dataset in ("cifar10", "cifar100"):
            manifest = _prepare_cifar(root, dataset, directory)
        elif dataset in HUMAN_KEYS:
            manifest = _prepare_human(root, dataset, directory)
        elif dataset == "fashionmnist":
            manifest = _prepare_fashion(root, directory)
        elif dataset == "pathmnist":
            manifest = _prepare_pathmnist(root, directory)
        elif dataset == "eurosat":
            manifest = _prepare_eurosat(root, directory)
        else:
            manifest = _prepare_speech(root, directory)
    print(f"{dataset}: ready ({manifest['rows']})", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and prepare FedLN datasets; omit --dataset to prepare all.")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--validate", action="store_true", help="Verify all prepared files without downloads")
    args = parser.parse_args(argv)
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        prepare_dataset(dataset, args.root, args.validate)


if __name__ == "__main__":
    main()
