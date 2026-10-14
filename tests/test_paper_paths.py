import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from types import SimpleNamespace


def fixture_images(root,name,shape=(32,32,3),classes=10):
    rng=np.random.default_rng(42)
    train=rng.integers(0,256,(32,*shape),dtype=np.uint8)
    test=rng.integers(0,256,(7,*shape),dtype=np.uint8)
    labels=np.arange(32)%classes; test_labels=np.arange(7)%classes
    np.savez(root/f'{name}.npz',train_images=train,train_labels=labels,test_images=test,test_labels=test_labels)
    return train,labels,test,test_labels


@pytest.mark.parametrize('name,shape,classes',[('fashionmnist',(28,28),10),('pathmnist',(28,28,3),9),('cifar100',(32,32,3),100)])
def test_added_vision_datasets_preserve_ids_and_class_space(tmp_path,name,shape,classes):
    from data.vision import load_vision
    fixture_images(tmp_path,name,shape,classes)
    seen=[]
    for cid in range(2):
        ds,n,count,labels,ids=load_vision(name,root=tmp_path,shard_id=cid,num_shards=2,batch_size=4,
                                         noise_lvl=0,noise_sparsity=0,shuffle=False)
        assert n==classes and labels.shape==(count,classes)
        seen.extend(ids.ravel())
        np.testing.assert_array_equal(np.concatenate([y.numpy() for _,y in ds]),labels)
    np.testing.assert_array_equal(sorted(seen),np.arange(32))


def test_human_noise_uses_selected_train_annotations_and_clean_test(tmp_path):
    from data.vision import load_vision, load_vision_test
    _,clean,_,test_clean=fixture_images(tmp_path,'cifar10')
    human=(clean+1)%10
    path=tmp_path/'human.npz'; np.savez(path,clean_label=clean,worse_label=human)
    _,_,_,actual,ids=load_vision('cifar10n',root=tmp_path,human_labels=path,human_key='worse_label',
                                num_shards=1,noise_lvl=.9,noise_sparsity=.9,shuffle=False)
    np.testing.assert_array_equal(actual.argmax(-1),human[ids.ravel()])
    ds,_,_=load_vision_test('cifar10n',root=tmp_path)
    np.testing.assert_array_equal(np.concatenate([y.numpy().argmax(-1) for _,y in ds]),test_clean)
    np.savez(path,clean_label=clean[::-1],worse_label=human)
    with pytest.raises(ValueError,match='order'):
        load_vision('cifar10n',root=tmp_path,human_labels=path,human_key='worse_label')


def test_optimizer_defaults_match_paper():
    from utils.optimizers import resolve_optimizer, make_optimizer
    for dataset,kind,lr,momentum in [('cifar10','sgd',.1,.9),('fashionmnist','sgd',.1,.9),
                                     ('pathmnist','adam',.001,0),('spcm','adam',.001,0)]:
        args=resolve_optimizer(SimpleNamespace(dataset_name=dataset,optimizer='auto',lr=None,momentum=.9))
        assert args.optimizer==kind and args.lr==lr
        optimizer=make_optimizer(vars(args))
        assert float(optimizer.learning_rate.numpy())==pytest.approx(lr)
        if kind=='sgd': assert optimizer.momentum==momentum


def test_audio_sidecar_order_is_checked(tmp_path,monkeypatch):
    from test_audio_data import write_records
    from data.spcm import load_raw_data
    monkeypatch.setenv('TFDS_DATA_DIR',str(tmp_path))
    write_records(tmp_path,'train',0,1,[0,1])
    sidecar=next((tmp_path/'speech_commands').glob('*.npz'))
    np.savez(sidecar,file_id=[1,0],label=[1,0])
    with pytest.raises(ValueError,match='order'):
        load_raw_data('train')


def test_feature_artifact_reorders_by_identity_and_rejects_source_mismatch(tmp_path):
    from utils.features import save_features,load_feature_rows
    save_features(tmp_path/'fixture.npy',[[9,8],[1,2]], [1,0],dataset='fixture',source_sha256='source',teacher={'name':'test'})
    np.testing.assert_array_equal(load_feature_rows(tmp_path,'fixture',[0,1],'source'),[[1,2],[9,8]])
    with pytest.raises(ValueError,match='source digest'):
        load_feature_rows(tmp_path,'fixture',[0],'different')
    np.save(tmp_path/'fixture.ids.npy',[1,0])
    with pytest.raises(ValueError,match='identity'):
        load_feature_rows(tmp_path,'fixture',[0],'source')


def test_global_logit_distillation_matches_paper_forward_kl():
    import tensorflow as tf
    from utils.distiller import forward_kl, GlobalLogitDistiller
    student=tf.constant([[2.,-.5],[0.,1.]])
    teacher=tf.constant([[.1,1.2],[2.,-1.]])
    p=tf.nn.softmax(student/2).numpy();q=tf.nn.softmax(teacher/2).numpy()
    expected=4*np.mean(np.sum(p*np.log(p/q),axis=-1))
    assert float(forward_kl(student,teacher,2))==pytest.approx(expected)
    reverse=4*np.mean(np.sum(q*np.log(q/p),axis=-1))
    assert abs(expected-reverse)>1e-3
    inputs=tf.keras.Input((2,)); model=tf.keras.Model(inputs,tf.keras.layers.Dense(2)(inputs))
    fixed=tf.keras.models.clone_model(model);fixed.set_weights(model.get_weights())
    before=[v.copy() for v in fixed.get_weights()]
    wrapped=GlobalLogitDistiller(model,fixed,beta=10,temperature=2)
    wrapped.compile(tf.keras.optimizers.SGD(.1),tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    [tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
    wrapped.train_step((tf.ones((2,2)),tf.eye(2)))
    for old,new in zip(before,fixed.get_weights()):np.testing.assert_array_equal(old,new)
    assert any(not np.array_equal(a,b) for a,b in zip(before,model.get_weights()))


def test_audio_export_assigns_dense_ids_and_retains_tfds_identity(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('prepare_audio',Path(__file__).resolve().parents[1]/'prepare_audio.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    examples=[dict(audio=np.arange(16000,dtype=np.int16),label=i%12,tfds_id=f'train-{i}'.encode()) for i in range(5)]
    module.export_split(examples,tmp_path,'train',2)
    from data.spcm import load_raw_data
    monkeypatch.setenv('TFDS_DATA_DIR',str(tmp_path))
    (ids,labels,paths),classes=load_raw_data('train')
    assert classes==12
    np.testing.assert_array_equal(ids,[0,2,4,1,3])
    np.testing.assert_array_equal(labels,[0,2,4,1,3])
    with np.load(paths[0]+'.npz',allow_pickle=False) as f:
        assert f['source_id'].tolist()==['train-0','train-2','train-4']
    with pytest.raises(FileExistsError):module.export_split(examples,tmp_path,'train',2)


@pytest.mark.parametrize('dataset,shape,classes',[('fashionmnist',(28,28),10),('pathmnist',(28,28,3),9),('cifar10n',(32,32,3),10)])
def test_added_dataset_actual_round(tmp_path,dataset,shape,classes):
    import subprocess,json
    base='cifar10' if dataset=='cifar10n' else dataset
    _,labels,_,_=fixture_images(tmp_path,base,shape,classes)
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/'run.py'),'--method','fedavg','--dataset_name',dataset,
             '--data_root',str(tmp_path),'--output',str(tmp_path/'run'),'--num_clients','2',
             '--num_rounds','1','--batch_size','8','--noisy_frac','0']
    if dataset=='cifar10n':
        np.savez(tmp_path/'human.npz',clean_label=labels,worse_label=(labels+1)%classes)
        command+=['--human_labels',str(tmp_path/'human.npz'),'--human_noise_key','worse_label']
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/'run'/'results.json').read_text())
    assert report['rounds'][0]['parameters_changed'] and report['rounds'][0]['examples']==32
