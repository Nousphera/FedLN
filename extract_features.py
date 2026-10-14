"""Extract authentic CLIP ViT-B/32 or TRILLsson3 embeddings with stable row IDs."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
from utils.features import array_digest,save_features,source_digest

CLIP_REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
TRILLSSON_HANDLE='https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1'


def extract_clip(input_path,batch_size=16,local_files_only=False):
    import torch
    from PIL import Image
    from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
    with np.load(input_path,allow_pickle=False) as file:
        images=file['images'];ids=file['ids'];dataset=str(file['dataset']);digest=str(file['source_sha256'])
    if digest!=array_digest(images):raise ValueError('Prepared image content changed')
    model=CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32',revision=CLIP_REVISION,local_files_only=local_files_only)
    processor=CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32',revision=CLIP_REVISION,local_files_only=local_files_only)
    model.eval();torch.set_num_threads(1)
    output=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            batch=[Image.fromarray(image.squeeze() if image.shape[-1:]==(1,) else image).convert('RGB') for image in images[start:start+batch_size]]
            pixel_values=processor(images=batch,return_tensors='pt')['pixel_values']
            output.append(model(pixel_values=pixel_values).image_embeds.cpu().numpy())
    teacher=dict(name='openai/clip-vit-base-patch32',revision=CLIP_REVISION,output='image_embeds',
                 l2_normalized=False,preprocessing='official CLIPImageProcessor; RGB conversion',
                 transformers=importlib.metadata.version('transformers'),torch=torch.__version__)
    return np.concatenate(output),ids,dataset,digest,teacher


def extract_trillsson(root,checkpoint=None,cache_dir=None):
    import tensorflow as tf
    from data.spcm import load_raw_data
    from utils.audio_utils import read_tf_record
    os.environ['TFDS_DATA_DIR']=str(root)
    if cache_dir:os.environ['TFHUB_CACHE_DIR']=str(cache_dir)
    if checkpoint is None:
        import tensorflow_hub as hub
        checkpoint=Path(hub.resolve(TRILLSSON_HANDLE))
    else:checkpoint=Path(checkpoint)
    model=tf.saved_model.load(str(checkpoint))
    (ids,_,paths),_=load_raw_data('train')  # Validates record/sidecar order.
    output=[];observed=[]
    for record in tf.data.TFRecordDataset(paths,compression_type='ZLIB',num_parallel_reads=1):
        waveform,_,identity=read_tf_record(record)
        tf.debugging.assert_less_equal(tf.reduce_max(waveform),tf.constant(32767,tf.int64))
        tf.debugging.assert_greater_equal(tf.reduce_min(waveform),tf.constant(-32768,tf.int64))
        waveform=tf.cast(waveform,tf.float32)/32768.
        # Official SavedModel owns its feature frontend and context padding.
        embedding=model(waveform[None,:])['embedding'].numpy()
        if embedding.ndim!=2 or embedding.shape!=(1,1024):raise ValueError(f'Unexpected TRILLsson output {embedding.shape}')
        output.append(embedding[0]);observed.append(int(identity.numpy()))
    np.testing.assert_array_equal(observed,ids)
    weight_hash=hashlib.sha256()
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            weight_hash.update(str(path.relative_to(checkpoint)).encode())
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):weight_hash.update(chunk)
    teacher=dict(name='TRILLsson3 (EfficientNetV2-B3)',handle=TRILLSSON_HANDLE,
                 artifact_sha256=weight_hash.hexdigest(),sample_rate=16000,pcm_scale=32768,
                 preprocessing='official SavedModel frontend; complete waveform; no external L2 normalization',
                 tensorflow=tf.__version__)
    return np.asarray(output),ids,'spcm',source_digest('spcm'),teacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',choices=['clip','trillsson3'],required=True)
    parser.add_argument('--input',type=Path,help='Prepared images NPZ for CLIP')
    parser.add_argument('--tfds-data-dir',type=Path,help='Prepared SpeechCommands root for TRILLsson')
    parser.add_argument('--output',type=Path,required=True,help='Dataset-named .npy file, e.g. features/cifar10.npy')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--checkpoint',type=Path,help='Local verified TRILLsson3 SavedModel (optional)')
    parser.add_argument('--cache-dir',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists')
    if args.batch_size<1:parser.error('batch-size must be positive')
    if args.teacher=='clip':
        if args.input is None:parser.error('--input is required for CLIP')
        result=extract_clip(args.input,args.batch_size,args.local_files_only)
    else:
        if args.tfds_data_dir is None:parser.error('--tfds-data-dir is required for TRILLsson')
        result=extract_trillsson(args.tfds_data_dir,args.checkpoint,args.cache_dir)
    features,ids,dataset,digest,teacher=result
    if args.output.name!=f'{dataset}.npy':parser.error(f'Output must be named {dataset}.npy')
    manifest=save_features(args.output,features,ids,dataset=dataset,source_sha256=digest,teacher=teacher)
    print(f"Extracted {manifest['rows']} x {manifest['width']} authentic teacher features")


if __name__=='__main__':main()
