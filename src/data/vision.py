"""Paper vision datasets with canonical, split-local row identities.

Optional NPZ inputs make preparation auditable and permit bounded offline tests.
CIFAR human labels must be in Keras/Python-pickle order, not TFDS order.
"""
from pathlib import Path
import os
import functools
import numpy as np
import tensorflow as tf
from utils import image_utils
from utils.shard import shard_data
from utils.noise import inject_synthetic_noise_to_labels

CLASSES={'cifar10':10,'cifar10n':10,'cifar100':100,'cifar100n':100,'fashionmnist':10,'pathmnist':9}
ALIASES={'cifar10n':'cifar10','cifar100n':'cifar100'}


def raw_vision(name,root=None):
    base=ALIASES.get(name,name)
    root=Path(root or os.environ.get('FEDLN_DATA_ROOT','data'))
    path=root/f'{base}.npz'
    if path.exists():
        from prepared_arrays import read_arrays
        data = read_arrays(path, base)
        arrays = tuple(data[k] for k in ('train_images','train_labels','test_images','test_labels'))
    elif base=='pathmnist':
        raise FileNotFoundError(f'{path}: obtain the official PathMNIST NPZ; see docs/PAPER_DATA.md')
    else:
        loader={'cifar10':tf.keras.datasets.cifar10,'cifar100':tf.keras.datasets.cifar100,
                'fashionmnist':tf.keras.datasets.fashion_mnist}[base]
        train,test=loader.load_data()
        arrays=(*train,*test)
    train_x,train_y,test_x,test_y=arrays
    train_y=np.asarray(train_y).reshape(-1);test_y=np.asarray(test_y).reshape(-1)
    for x,y in [(train_x,train_y),(test_x,test_y)]:
        if len(x)!=len(y) or x.dtype!=np.uint8 or not len(x):
            raise ValueError('Images must be nonempty uint8 arrays with one integer label per row')
        if not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>=CLASSES[name])):
            raise ValueError('Labels outside the declared dataset class space')
    return train_x,train_y,test_x,test_y


def human_annotations(path,key,clean,name):
    allowed={'cifar10n':{'worse_label','aggre_label','random_label1','random_label2','random_label3'},
             'cifar100n':{'noisy_label'}}[name]
    if not path or key not in allowed:
        raise ValueError(f'{name} requires --human_labels NPZ and explicit --human_noise_key in {sorted(allowed)}')
    with np.load(path,allow_pickle=False) as labels:
        original=np.asarray(labels['clean_label']).reshape(-1)
        noisy=np.asarray(labels[key]).reshape(-1)
    if not np.array_equal(original,clean):
        raise ValueError('Human annotation clean_label does not match training order; TFDS order is unsupported')
    if noisy.shape!=clean.shape or not np.issubdtype(noisy.dtype,np.integer) or np.any((noisy<0)|(noisy>=CLASSES[name])):
        raise ValueError('Human annotations have invalid shape or labels')
    return noisy


def transforms(name,train):
    base=ALIASES.get(name,name)
    suffix='cifar' if base.startswith('cifar') else 'fmnist' if base=='fashionmnist' else 'pathmnist'
    return getattr(image_utils,('train_prep_' if train else 'valid_prep_')+suffix),getattr(image_utils,'cutout_'+suffix)


def load_vision(name,shard_id=0,num_shards=None,batch_size=128,seed=42,noisy_clients_frac=.6,
                noise_lvl=None,noise_sparsity=None,corrected_labels=None,shuffle=True,
                root=None,human_labels=None,human_key=None):
    images,labels,_,_=raw_vision(name,root)
    classes=CLASSES[name]
    human=name in ALIASES
    if human: labels=human_annotations(human_labels,human_key,labels,name)
    ids=np.arange(len(images))[:,None]
    if num_shards is not None:
        (images,labels),ids=shard_data((images,labels),id=int(shard_id),num_shards=num_shards,seed=seed)
    if not len(images):raise ValueError('Partition contains no examples; reduce num_clients')
    labels=np.eye(classes,dtype=np.float32)[labels]
    if corrected_labels is not None:
        labels=np.asarray(corrected_labels,dtype=np.float32)
        if labels.shape!=(len(images),classes):raise ValueError('Corrected label shape differs from client partition')
    if not human and corrected_labels is None and noise_lvl is not None and noise_sparsity is not None:
        rng=np.random.RandomState(seed)
        noisy=rng.choice(num_shards,size=min(num_shards,int(round(noisy_clients_frac*num_shards))),replace=False) if num_shards else [shard_id]
        if int(shard_id) in noisy:
            labels=inject_synthetic_noise_to_labels(shard_id,labels,classes,level=noise_lvl,sparsity=noise_sparsity,seed=seed,theshold=5e-2)
    preprocess,cutout=transforms(name,True)
    ds=tf.data.Dataset.from_tensor_slices((images,labels)).map(preprocess)
    if shuffle:ds=ds.shuffle(min(len(images),10000),seed=seed,reshuffle_each_iteration=True)
    ds=ds.batch(batch_size).map(cutout).prefetch(1)
    return ds,classes,len(images),labels,ids


def load_vision_test(name,batch_size=128,root=None):
    _,_,images,labels=raw_vision(name,root)
    labels=np.eye(CLASSES[name],dtype=np.float32)[labels]
    preprocess,_=transforms(name,False)
    ds=tf.data.Dataset.from_tensor_slices((images,labels)).map(preprocess).batch(batch_size).prefetch(1)
    return ds,CLASSES[name],len(labels)
