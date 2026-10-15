"""Download pretrained teachers and prepare canonical training embeddings."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fedln.data import DATASETS, load_data
from fedln.teachers import load_teacher, prepare_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='cifar10')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--model-root', type=Path, default=Path('checkpoints'))
    parser.add_argument('--feature-root', type=Path, default=Path('features'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--human-noise-key')
    parser.add_argument('--teacher-only', action='store_true', help='Download weights without extracting features')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    import torch
    torch.set_num_threads(2)
    if args.teacher_only:
        load_teacher(args.dataset, args.model_root, args.device)
        print(f'{args.dataset}: pretrained teacher ready in {args.model_root}')
        return
    from fedln.preparation import verify_dataset
    verify_dataset(args.dataset, args.data_root)
    bundle = load_data(args.dataset, args.data_root, training=False, human_noise_key=args.human_noise_key)
    prepare_features(args.dataset, bundle, args.model_root, args.feature_root, args.batch_size, args.device)


if __name__ == '__main__':
    main()
