"""Run original FedLN clients and strategies sequentially with stock Flower transport."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
METHODS = {'nnc': 'fedln/fedln__nnc.py', 'akd': 'fedln/fedln__akd.py',
           'na_fedavg': 'fedln/fedln__nafedavg.py', 'fedavg': 'baselines/fedavg__standard.py',
           'label_smoothing': 'baselines/fedavg__smooth.py',
           'bi_tempered_loss': 'baselines/fedavg__loss.py',
           'confidence_learning': 'baselines/fedavg__cl.py'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='nnc')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path('runs/fedln'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--method-help', action='store_true', help='Show the original method arguments')
    options, forwarded = parser.parse_known_args(argv)
    if options.method_help: forwarded.append('--help')
    output = options.output.resolve()
    if not options.method_help and output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    os.environ['FEDLN_DEVICE'] = options.device
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if options.device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(options.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(options.threads)
    sys.path.insert(0, str(ROOT / 'src'))
    sys.argv = [str(ROOT / 'src' / METHODS[options.method]), *forwarded]
    spec = importlib.util.spec_from_file_location('fedln_experiment', sys.argv[0])
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    args = experiment.args
    for name in ('num_clients', 'num_rounds', 'batch_size', 'train_epochs'):
        if getattr(args, name) <= 0: parser.error(f'{name} must be positive')
    if not 0 < args.participation_rate <= 1: parser.error('participation_rate must be in (0,1]')
    import numpy as np
    import tensorflow as tf
    import flwr as fl
    from flwr.common import serde
    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    report = {'method': options.method, 'paper_reproduction': False, 'backend': 'sequential-stock-flower',
              'config': {**vars(args), **{k: str(v) if isinstance(v, Path) else v for k,v in vars(options).items()}},
              'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                          'tensorflow': tf.__version__, 'flower': fl.__version__, 'numpy': np.__version__}, 'rounds': []}
    # Files are owned by this invocation; no existing cache directory is deleted.
    with tempfile.TemporaryDirectory(prefix='cache-', dir=output) as cache:
        if hasattr(args, 'temp_dir'): args.temp_dir = cache
        if hasattr(args, 'embeddings_dir'):
            if args.dataset_name == 'synthetic':
                from data.synthetic import features
                args.embeddings_dir = cache
                np.save(Path(cache) / 'synthetic.npy', features(args.seed))
            else:
                path = Path(args.embeddings_dir)
                args.embeddings_dir = str(path.resolve() if path.is_absolute() else (ROOT / 'src' / path).resolve())
        server = experiment.create_server()
        strategy = server.strategy
        parameters = strategy.initial_parameters
        # configure_fit owns sampling, including all-client NA-FedAvg warmup.
        class LocalManager:
            def num_available(self): return args.num_clients
            def sample(self, num_clients, min_num_clients=None, criterion=None):
                from types import SimpleNamespace
                return [SimpleNamespace(cid=str(i)) for i in sorted(random.sample(range(args.num_clients), num_clients))]
        manager = LocalManager()
        for rnd in range(1, args.num_rounds + 1):
            before = fl.common.parameters_to_ndarrays(parameters)
            results = []
            for proxy, instruction in strategy.configure_fit(rnd, parameters, manager):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(args.seed + rnd * args.num_clients + int(proxy.cid))
                client = experiment.create_client(proxy.cid)
                instruction = serde.fit_ins_from_proto(serde.fit_ins_to_proto(instruction))
                response = client.to_client().fit(instruction)
                response = serde.fit_res_from_proto(serde.fit_res_to_proto(response))
                results.append((proxy, response))
            parameters, aggregation_metrics = strategy.aggregate_fit(rnd, results, [])
            if parameters is None: raise RuntimeError('Aggregation did not return model parameters')
            loss, metrics = strategy.evaluate(rnd, parameters)
            after = fl.common.parameters_to_ndarrays(parameters)
            if len(before) != len(after): raise RuntimeError('Client changed the global parameter structure')
            entry = {'round': rnd, 'clients': [p.cid for p, _ in results],
                     'examples': sum(r.num_examples for _, r in results),
                     'loss': float(loss), 'accuracy': float(metrics['accuracy']),
                     'parameters_changed': any(not np.array_equal(a,b) for a,b in zip(before,after)),
                     'aggregation_metrics': aggregation_metrics, 'client_metrics': [r.metrics for _,r in results]}
            report['rounds'].append(entry)
            (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps(entry), flush=True)
        np.savez_compressed(output / 'final-parameters.npz', *fl.common.parameters_to_ndarrays(parameters))
    return report


if __name__ == '__main__':
    main()
