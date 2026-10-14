"""Recovered prepared-TFRecord audio path; no corpus download."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest


def write_records(root, split, shard, total, labels):
    import tensorflow as tf
    folder=root/'speech_commands';folder.mkdir(exist_ok=True)
    path=folder/f'speech_commands-{split}.tfrecord-{shard:05d}-of-{total:05d}'
    ids=np.arange(len(labels))+shard*len(labels)
    with tf.io.TFRecordWriter(str(path), options='ZLIB') as writer:
        for identity,label in zip(ids,labels):
            wave=(np.sin(np.arange(16000)*(label+1)/100)*10000).astype(np.int64)
            fields={k:tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
                    for k,v in {'label':label,'shape':len(wave),'file_id':identity}.items()}
            for k,v in {'audio':tf.io.serialize_tensor(wave).numpy(),'speaker_id':b'fixture'}.items():
                fields[k]=tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
            writer.write(tf.train.Example(features=tf.train.Features(feature=fields)).SerializeToString())
    np.savez(str(path)+'.npz',file_id=ids,label=labels)


def test_recovered_audio_loader_preserves_ids_labels_and_counts(tmp_path, monkeypatch):
    from data.spcm import load_spcm, load_spcm_test
    monkeypatch.setenv('TFDS_DATA_DIR',str(tmp_path))
    write_records(tmp_path,'train',0,2,[0,1,0,1])
    write_records(tmp_path,'train',1,2,[2,3,2,3])
    write_records(tmp_path,'test',0,1,[0,1])
    seen=[]
    for cid in range(2):
        ds,classes,count,labels,ids=load_spcm(cid,2,batch_size=2,shuffle=False)
        actual=np.concatenate([y.numpy() for _,y in ds])
        np.testing.assert_array_equal(actual,labels)
        assert classes==12 and count==4
        seen.extend(ids.reshape(-1).tolist())
        corrected=np.eye(12)[np.full(count,5)]
        ds,*_=load_spcm(cid,2,batch_size=2,shuffle=False,corrected_labels=corrected)
        assert all(np.all(y.numpy().argmax(-1)==5) for _,y in ds)
    assert sorted(seen)==list(range(8))
    ds,_,count,_,ids=load_spcm(0,1,batch_size=2,shuffle=False)
    assert count==8 and sum(len(y) for _,y in ds)==8
    ds,classes,count=load_spcm_test()
    assert count==2 and classes==12
    assert next(iter(ds))[0].shape==(1,98,64,1)


@pytest.mark.parametrize('method', ['nnc','akd'])
def test_audio_methods_train_from_prepared_records(tmp_path, method):
    import json,subprocess
    for shard in range(2):write_records(tmp_path,'train',shard,2,[0,1]*4)
    write_records(tmp_path,'test',0,1,[0,1])
    features=np.eye(2)[[0,1]*8].astype(np.float32)
    np.save(tmp_path/'spcm.npy',features)
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'run.py'),
        '--method',method,'--dataset_name','spcm','--model_name','cnn',
        '--num_clients','2','--num_rounds','2','--train_epochs','1','--batch_size','4',
        '--knn_neighbors','1','--embeddings_dir',str(tmp_path),'--noisy_frac','0',
        '--output',str(tmp_path/'run')]
    if method=='akd':command+=['--embeddings_dims','2']
    env={**os.environ,'TFDS_DATA_DIR':str(tmp_path)}
    result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    rows=json.loads((tmp_path/'run/results.json').read_text())['rounds']
    assert all(r['examples']==16 and r['parameters_changed'] for r in rows)
