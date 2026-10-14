"""Maintained version of the recovered FedLN prepared-TFRecord audio loader.

Original: HomeLab FedLN/data/spcm.py; see docs/RECOVERED_DATA.md.
"""
import functools
import os
from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from utils.audio_utils import train_prep_spcm, valid_prep_spcm, read_tf_record
from utils.noise import inject_synthetic_noise_to_labels


def _record_paths(split):
    if 'TFDS_DATA_DIR' not in os.environ:
        raise ValueError('Set TFDS_DATA_DIR to the prepared dataset root')
    root=Path(os.environ['TFDS_DATA_DIR'])/'speech_commands'
    paths=list(root.glob(f'speech_commands-{split}.tfrecord-*-of-*'))
    paths=[p for p in paths if not p.name.endswith('.npz')]
    paths.sort(key=lambda p:int(re.search(r'tfrecord-(\d+)-of-',p.name).group(1)))
    if not paths: raise FileNotFoundError(f'No prepared {split} TFRecords under {root}')
    return paths


def load_raw_data(split, shards='all'):
    paths=_record_paths(split)
    if not isinstance(shards,str): paths=[paths[int(i)] for i in np.asarray(shards).reshape(-1)]
    ids=[];labels=[]
    for path in paths:
        with np.load(str(path)+'.npz',allow_pickle=False) as sidecar:
            ids.append(np.asarray(sidecar['file_id'],dtype=np.int64).reshape(-1))
            labels.append(np.asarray(sidecar['label'],dtype=np.int64).reshape(-1))
    if not ids: raise ValueError('Each client needs at least one prepared TFRecord shard')
    ids=np.concatenate(ids);labels=np.concatenate(labels)
    if len(ids)!=len(labels) or len(np.unique(ids))!=len(ids) or np.any(ids<0):
        raise ValueError('Sidecar IDs must be unique nonnegative feature-row indices')
    if np.any((labels<0)|(labels>=12)): raise ValueError('SpeechCommands labels must be in [0,12)')
    return (ids,labels,[str(p) for p in paths]),12


def _dataset(paths, ids, labels):
    table=tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(ids,np.argmax(labels,axis=-1).astype(np.int64)),default_value=-1)
    def mapped(serialized):
        audio,_,identity=read_tf_record(serialized)
        label=table.lookup(identity)
        with tf.control_dependencies([tf.debugging.assert_greater_equal(label,tf.constant(0,tf.int64))]):
            return audio,tf.identity(label)
    # Sidecar and record order must match; preserve shard order for embeddings.
    return tf.data.TFRecordDataset(paths,compression_type='ZLIB',num_parallel_reads=1).map(mapped)


def load_spcm(shard_id=0,num_shards=None,batch_size=128,seed=42,noisy_clients_frac=.6,
              noise_lvl=None,noise_sparsity=None,corrected_labels=None,shuffle=True):
    shards='all'
    if num_shards is not None:
        if num_shards>len(_record_paths('train')): raise ValueError('Clients exceed prepared TFRecord shards')
        shards=np.array_split(np.random.RandomState(seed).permutation(len(_record_paths('train'))),num_shards)[int(shard_id)]
    (ids,raw_labels,paths),classes=load_raw_data('train',shards)
    labels=np.eye(classes,dtype=np.float32)[raw_labels]
    if corrected_labels is not None:
        labels=np.asarray(corrected_labels,dtype=np.float32)
        if labels.shape!=(len(ids),classes):raise ValueError('Corrected labels must match client rows and twelve classes')
    rng=np.random.RandomState(seed)
    noisy=rng.choice(num_shards,size=min(num_shards,int(round(noisy_clients_frac*num_shards))),replace=False) if num_shards else [shard_id]
    if noise_lvl is not None and noise_sparsity is not None and corrected_labels is None and int(shard_id) in noisy:
        labels=inject_synthetic_noise_to_labels(shard_id,labels,classes,level=noise_lvl,sparsity=noise_sparsity,seed=seed,theshold=5e-2)
    ds=_dataset(paths,ids,labels)
    if shuffle:ds=ds.shuffle(10000,seed=seed,reshuffle_each_iteration=True)
    ds=ds.map(functools.partial(train_prep_spcm,n_mels=64)).batch(batch_size).prefetch(1)
    return ds,classes,len(ids),labels,ids[:,None]


def load_spcm_test(batch_size=1):
    (ids,raw_labels,paths),classes=load_raw_data('test')
    ds=_dataset(paths,ids,np.eye(classes,dtype=np.float32)[raw_labels])
    # Preserve full evaluation duration; variable length requires single-item batches.
    ds=ds.map(functools.partial(valid_prep_spcm,n_mels=64)).batch(1).prefetch(1)
    return ds,classes,len(ids)
