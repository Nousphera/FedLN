"""Export canonical, unaugmented training images for an isolated CLIP environment."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
from utils.features import array_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['cifar10','cifar10n','cifar100','cifar100n','fashionmnist','pathmnist','eurosat'])
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists')
    if args.dataset=='eurosat':
        with np.load(args.data_root/'eurosat'/'eurosat.npz',allow_pickle=False) as f:images=f['images'][:21600]
    else:
        from data.vision import raw_vision
        images=raw_vision(args.dataset,args.data_root)[0]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:
        np.savez(f,images=images,ids=np.arange(len(images)),dataset=args.dataset,source_sha256=array_digest(images))
    print(f'Prepared {len(images)} unaugmented training rows; digest {array_digest(images)}')


if __name__=='__main__':main()
