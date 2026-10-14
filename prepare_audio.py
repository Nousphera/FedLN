"""Export TFDS Speech Commands 0.0.3 to the recovered FedLN waveform schema."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

LABELS=['down','go','left','no','off','on','right','stop','up','yes','_silence_','_unknown_']


def export_split(examples, root, split, shards=1):
    """Assign deterministic dense row IDs; retain source TFDS IDs in each sidecar."""
    import tensorflow as tf
    if shards < 1:raise ValueError('shards must be positive')
    directory=Path(root)/'speech_commands'
    directory.mkdir(parents=True,exist_ok=True)
    paths=[directory/f'speech_commands-{split}.tfrecord-{i:05d}-of-{shards:05d}' for i in range(shards)]
    if any(p.exists() or Path(str(p)+'.npz').exists() for p in paths):
        raise FileExistsError('Prepared output already exists')
    writers=[tf.io.TFRecordWriter(str(path),options='ZLIB') for path in paths]
    rows=[[] for _ in paths]; labels=[[] for _ in paths]; source_ids=[[] for _ in paths]
    seen=set()
    try:
        for identity,example in enumerate(examples):
            audio=np.asarray(example['audio'])
            label=int(example['label'])
            source_id=example['tfds_id']
            if isinstance(source_id,bytes):source_id=source_id.decode('utf-8')
            source_id=str(source_id)
            if source_id in seen:raise ValueError('Repeated source identity')
            seen.add(source_id)
            if audio.ndim!=1 or not len(audio) or audio.dtype!=np.int16:
                raise ValueError('Expected nonempty mono signed 16-bit PCM')
            if not 0<=label<12:raise ValueError('Expected twelve-class TFDS label')
            integers=lambda value:tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            blob=lambda value:tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
            features={'audio':blob(tf.io.serialize_tensor(tf.convert_to_tensor(audio,tf.int64)).numpy()),
                      'shape':integers([len(audio)]),'label':integers([label]),'file_id':integers([identity]),
                      'speaker_id':blob(b''),'source_id':blob(source_id.encode())}
            shard=identity%shards
            writers[shard].write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
            rows[shard].append(identity);labels[shard].append(label);source_ids[shard].append(source_id)
    finally:
        for writer in writers:writer.close()
    if any(not row for row in rows):raise ValueError('Not enough examples for the requested shard count')
    artifacts=[]
    for path,identities,targets,sources in zip(paths,rows,labels,source_ids):
        with Path(str(path)+'.npz').open('xb') as f:
            np.savez(f,file_id=np.asarray(identities,dtype=np.int64),label=np.asarray(targets,dtype=np.int64),
                     source_id=np.asarray(sources,dtype=np.str_))
        artifacts.append({'file':path.name,'rows':len(identities),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'rows':sum(map(len,rows)),'shards':artifacts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tfds-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New prepared dataset root')
    parser.add_argument('--train-shards',type=int,default=100)
    parser.add_argument('--download',action='store_true',help='Download/prepare the official corpus if missing')
    parser.add_argument('--limit-per-split',type=int,help='Explicit small-fixture limit; not a benchmark dataset')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output root already exists')
    if args.train_shards<1 or (args.limit_per_split is not None and args.limit_per_split<args.train_shards):
        parser.error('Positive shard count and at least that many training examples required')
    import tensorflow_datasets as tfds
    builder=tfds.builder('speech_commands:0.0.3',data_dir=str(args.tfds_cache))
    if args.download:builder.download_and_prepare()
    if builder.info.features['label'].names!=LABELS:raise ValueError('Unexpected TFDS label mapping')
    manifest={'schema_version':1,'dataset':'speech_commands','tfds_version':tfds.__version__,
              'builder_version':str(builder.version),'labels':LABELS,'sample_rate':16000,
              'partition':'round-robin record shards; loader assigns shards to clients',
              'speaker_ids_available':False,'paper_partition_recovered':False,
              'limit_per_split':args.limit_per_split,'splits':{}}
    for split in ['train','validation','test']:
        dataset=builder.as_dataset(split=split,shuffle_files=False,
            read_config=tfds.ReadConfig(add_tfds_id=True,interleave_cycle_length=1))
        if args.limit_per_split:dataset=dataset.take(args.limit_per_split)
        manifest['splits'][split]=export_split(tfds.as_numpy(dataset),args.output,split,
                                             args.train_shards if split=='train' else 1)
    (args.output/'speech_commands'/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({split:value['rows'] for split,value in manifest['splits'].items()}))


if __name__=='__main__':main()
