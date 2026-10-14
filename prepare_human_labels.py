"""Convert official CIFAR-N PyTorch annotation files to pickle-free NumPy input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def convert(source,output,dataset):
    import torch
    source,output=Path(source),Path(output)
    if output.exists() or output.with_suffix('.manifest.json').exists():raise FileExistsError('Annotation output exists')
    # Official files contain only NumPy int64 arrays. Never enable unrestricted pickle loading.
    reconstruct=np.core.multiarray._reconstruct
    allowed=[(reconstruct,'numpy.core.multiarray._reconstruct'),
             (reconstruct,'numpy._core.multiarray._reconstruct'),np.ndarray,np.dtype,np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        data=torch.load(source,map_location='cpu',weights_only=True)
    keys=['clean_label','aggre_label','worse_label','random_label1','random_label2','random_label3'] if dataset=='cifar10n' else ['clean_label','noisy_label']
    classes=10 if dataset=='cifar10n' else 100
    result={}
    for key in keys:
        value=np.asarray(data[key])
        if value.shape!=(50000,) or value.dtype!=np.int64 or np.any((value<0)|(value>=classes)):
            raise ValueError(f'{key}: expected 50,000 official int64 class IDs')
        result[key]=value
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as file:np.savez(file,**result)
    manifest={'dataset':dataset,'source_name':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'source_project':'https://github.com/UCSC-REAL/cifar-10-100n','format':'official Python/CIFAR training order',
              'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'annotation_keys':keys}
    with output.with_suffix('.manifest.json').open('x') as file:json.dump(manifest,file,indent=2)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['cifar10n','cifar100n'],required=True)
    args=parser.parse_args()
    print(json.dumps(convert(args.input,args.output,args.dataset),indent=2))


if __name__=='__main__':main()
