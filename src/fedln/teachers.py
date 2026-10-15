"""Pinned pretrained teachers and row-aligned feature preparation."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np
import torch

CLIP_MODEL = 'openai/clip-vit-base-patch32'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CLIP_SHA256 = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
TRILLSSON_URL = 'https://github.com/Nousphera/FedLN/releases/download/v0.2.0/trillsson3.pt'
TRILLSSON_SHA256 = '256fc6a004f1ff285b86941c7b0f519db1991c0856acd13dd01b15ef91d9cdda'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _cached_digest(path, size, mtime_ns, ctime_ns):
    return file_digest(path)


def verified_digest(path):
    """Hash unchanged files once per worker process."""
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _teacher_info(dataset):
    if dataset == 'spcm':
        return dict(name='TRILLsson3', checkpoint_sha256=TRILLSSON_SHA256,
                    output='embedding', width=1024, sample_rate=16000,
                    l2_normalized=False)
    return dict(name=CLIP_MODEL, revision=CLIP_REVISION, checkpoint_sha256=CLIP_SHA256, output='image_embeds',
                width=512, processor_backend='pil', l2_normalized=False)


def _feature_identity(dataset, bundle):
    ids = np.asarray(bundle.train.ids, dtype='<i8')
    if not np.array_equal(ids, np.arange(len(bundle.train), dtype=np.int64)):
        raise ValueError('Teacher preparation requires canonical training row IDs')
    return dict(schema=1, dataset=dataset, source_sha256=bundle.fingerprint,
                rows=len(ids), ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                teacher=_teacher_info(dataset))


def load_features(dataset, root, bundle):
    """Load a verified float32 feature matrix in canonical training-row order."""
    path = Path(root) / f'{dataset}.npy'
    manifest_path = path.with_suffix('.json')
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'Prepare embeddings first: python scripts/prepare_model.py --dataset {dataset}')
    manifest = json.loads(manifest_path.read_text())
    expected = _feature_identity(dataset, bundle)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Features do not match this dataset or teacher. Remove {path} and {manifest_path}, then rerun prepare_model.py')
    if verified_digest(path) != manifest.get('features_sha256'):
        raise ValueError(f'Feature checksum mismatch: {path}')
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (expected['rows'], expected['teacher']['width']):
        raise ValueError(f'Invalid feature matrix shape or dtype: {path}')
    if not np.isfinite(matrix).all():
        raise ValueError(f'Nonfinite teacher features: {path}')
    return matrix


def _download_trillsson(root):
    path = Path(root) / 'trillsson3.pt'
    if path.exists():
        if verified_digest(path) != TRILLSSON_SHA256:
            raise ValueError(f'TRILLsson3 checksum mismatch; remove the damaged checkpoint: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.part')
    try:
        with urlopen(TRILLSSON_URL, timeout=60) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if file_digest(temporary) != TRILLSSON_SHA256:
            raise ValueError('Downloaded TRILLsson3 checkpoint failed its SHA256 check')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _clip_snapshot(root):
    """Reuse the Hugging Face cache, with local model files under checkpoints/."""
    from huggingface_hub import hf_hub_download
    directory = Path(root) / 'clip-vit-b32'
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'):
        source = Path(hf_hub_download(CLIP_MODEL, name, revision=CLIP_REVISION))
        if name == 'pytorch_model.bin' and verified_digest(source) != CLIP_SHA256:
            raise ValueError('Cached CLIP weights failed their SHA256 check; remove the damaged Hugging Face cache file')
        destination = directory / name
        if destination.exists():
            if destination.samefile(source):
                continue
            if file_digest(destination) == file_digest(source):
                continue
            raise ValueError(f'CLIP checkpoint differs from its pinned revision: {destination}')
        temporary = destination.with_suffix(destination.suffix + '.part')
        temporary.unlink(missing_ok=True)
        try:
            os.link(source.resolve(), temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    return directory


def load_teacher(dataset, root='checkpoints', device='cpu'):
    """Download verified weights and load the paper's frozen feature extractor."""
    if dataset == 'spcm':
        from .trillsson import Trillsson3
        path = _download_trillsson(root)
        model = Trillsson3()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        processor = None
    else:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        directory = _clip_snapshot(root)
        model = CLIPVisionModelWithProjection.from_pretrained(directory, local_files_only=True)
        # Explicit PIL processor preserves original resize/crop/normalization.
        processor = CLIPImageProcessor.from_pretrained(directory, local_files_only=True)
    model.eval().requires_grad_(False).to(device)
    return model, processor


def _extract_batch(dataset, rows, model, processor, device):
    with torch.inference_mode():
        if dataset != 'spcm':
            pixels = processor(images=[row.convert('RGB') for row in rows],
                               return_tensors='pt')['pixel_values'].to(device)
            output = model(pixel_values=pixels).image_embeds
        else:
            # Only equal-length waveforms share a batch; external zero padding
            # would alter the teacher's native symmetric padding and contexts.
            groups = {}
            for index, row in enumerate(rows):
                groups.setdefault(len(row), []).append(index)
            output = torch.empty((len(rows), 1024), device=device)
            for indices in groups.values():
                waveforms = torch.stack([rows[i] for i in indices]).to(device)
                output[indices] = model(waveforms)
    matrix = output.float().cpu().numpy()
    if not np.isfinite(matrix).all():
        raise ValueError('The pretrained teacher produced nonfinite embeddings')
    return matrix


def prepare_features(dataset, bundle, model_root='checkpoints', feature_root='features',
                     batch_size=16, device='cpu'):
    """Extract unaugmented features, resuming completed batches after interruption."""
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    root = Path(feature_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{dataset}.npy'
    manifest_path = path.with_suffix('.json')
    identity = _feature_identity(dataset, bundle)
    if path.exists() and manifest_path.exists():
        load_features(dataset, root, bundle)
        print(f'{dataset}: verified {identity["rows"]} existing embeddings')
        return path
    partial = root / f'{dataset}.partial.npy'
    progress_path = root / f'{dataset}.partial.json'
    width = identity['teacher']['width']
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError(f'Incomplete features belong to a different dataset; remove {partial} and {progress_path}')
        start = progress.get('completed', 0)
        if type(start) != int or not 0 <= start <= identity['rows']:
            raise ValueError(f'Invalid feature progress: {progress_path}')
        features = np.load(partial, mmap_mode='r+', allow_pickle=False)
        if features.shape != (identity['rows'], width) or features.dtype != np.float32:
            raise ValueError(f'Invalid incomplete feature matrix: {partial}')
    else:
        start = 0
        features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                            shape=(identity['rows'], width))
        _write_json(progress_path, {**identity, 'completed': 0})
    model, processor = load_teacher(dataset, model_root, device)
    for offset in range(start, identity['rows'], batch_size):
        stop = min(offset + batch_size, identity['rows'])
        rows = [bundle.train.raw(i) for i in range(offset, stop)]
        batch = _extract_batch(dataset, rows, model, processor, device)
        if batch.shape != (stop - offset, width):
            raise ValueError(f'Unexpected teacher output shape: {batch.shape}')
        features[offset:stop] = batch
        features.flush()
        _write_json(progress_path, {**identity, 'completed': stop})
        if stop == identity['rows'] or offset // batch_size % 50 == 0:
            print(f'{dataset}: {stop}/{identity["rows"]} embeddings', flush=True)
    features.flush()
    del features
    partial.replace(path)
    _write_json(manifest_path, {**identity, 'features_sha256': file_digest(path),
                               'torch': torch.__version__,
                               'transformers': importlib.metadata.version('transformers')})
    progress_path.unlink(missing_ok=True)
    load_features(dataset, root, bundle)
    return path
