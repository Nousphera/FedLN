"""Verifiable canonical-row embedding artifacts; legacy arrays remain explicit."""
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np


def array_digest(array):
    array=np.ascontiguousarray(array)
    digest=hashlib.sha256()
    digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def source_digest(dataset,root=None):
    if dataset=='spcm':
        from data.spcm import _record_paths
        digest=hashlib.sha256()
        for path in _record_paths('train'):
            for item in (path,Path(str(path)+'.npz')):
                digest.update(item.name.encode())
                with item.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()
    if dataset=='eurosat':
        from data.eurosat import eurosat_data_path
        with np.load(eurosat_data_path(),allow_pickle=False) as data:return array_digest(data['images'][:21600])
    from data.vision import raw_vision
    return array_digest(raw_vision(dataset,root)[0])


def save_features(path,features,ids,*,dataset,source_sha256,teacher):
    path=Path(path);ids=np.asarray(ids,dtype=np.int64).reshape(-1)
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or len(features)!=len(ids) or not np.isfinite(features).all():
        raise ValueError('Features must be a finite matrix aligned with IDs')
    if not np.array_equal(np.sort(ids),np.arange(len(ids))):
        raise ValueError('Extraction IDs must cover each canonical row exactly once')
    ordered=np.empty_like(features);ordered[ids]=features
    ids_path=path.with_suffix('.ids.npy');manifest_path=path.with_suffix('.manifest.json')
    if any(p.exists() for p in (path,ids_path,manifest_path)):raise FileExistsError('Feature output exists')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.save(f,ordered,allow_pickle=False)
    with ids_path.open('xb') as f:np.save(f,np.arange(len(ids)),allow_pickle=False)
    manifest=dict(schema=1,dataset=dataset,split='train',rows=len(ids),width=features.shape[1],
                  source_sha256=source_sha256,features_sha256=array_digest(ordered),teacher=teacher)
    with manifest_path.open('x') as f:json.dump(manifest,f,indent=2)
    return manifest


def load_feature_rows(directory,dataset,ids,expected_source=None):
    path=Path(directory)/f'{dataset}.npy'
    features=np.load(path,allow_pickle=False,mmap_mode='r')
    ids=np.asarray(ids,dtype=np.int64).reshape(-1)
    if features.ndim!=2 or np.any(ids<0) or np.any(ids>=len(features)):
        raise ValueError('Embedding matrix does not cover client sample IDs')
    manifest_path=path.with_suffix('.manifest.json')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        canonical=np.load(path.with_suffix('.ids.npy'),allow_pickle=False)
        if manifest.get('dataset')!=dataset or manifest.get('split')!='train':raise ValueError('Embedding dataset/split mismatch')
        if not np.array_equal(canonical,np.arange(len(features))):raise ValueError('Embedding row identity mismatch')
        if manifest.get('features_sha256')!=array_digest(features):raise ValueError('Embedding content digest mismatch')
        if expected_source is not None and manifest.get('source_sha256')!=expected_source:
            raise ValueError('Embedding source digest differs from current dataset order/content')
    else:
        warnings.warn(f'{dataset}: legacy feature array has no identity manifest; alignment is unverified',RuntimeWarning)
    selected=np.asarray(features[ids],dtype=np.float32)
    if not np.isfinite(selected).all():raise ValueError('Nonfinite embeddings')
    return selected
