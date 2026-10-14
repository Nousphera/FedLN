"""Positive fractional participation must schedule an actual client."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import pytest


@pytest.mark.parametrize('method', ['fedavg', 'na_fedavg'])
def test_tiny_positive_participation_schedules_one_client(monkeypatch, method):
    import tensorflow as tf
    from utils.flwr_server import _Server
    monkeypatch.setenv('FEDLN_DEVICE', 'cpu')
    dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((2, 4)), tf.eye(2))).batch(2)
    def model_loader(input_shape, num_classes):
        x = tf.keras.Input(input_shape)
        return tf.keras.Model(x, tf.keras.layers.Dense(num_classes)(x))
    data_loader = lambda: (dataset, 2, 2)
    if method == 'fedavg':
        server = _Server(model_loader, data_loader, num_rounds=1, num_clients=2,
                         participation=.01, train_epochs=1, lr=.001)
    else:
        monkeypatch.setattr(sys, 'argv', ['test'])
        path = Path(__file__).resolve().parents[1]/'src/fedln/fedln__nafedavg.py'
        spec = importlib.util.spec_from_file_location('participation_na', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.args.num_clients = 2
        module.args.participation_rate = .01
        monkeypatch.setattr(module, 'load_available_models', lambda: {module.args.model_name: model_loader})
        monkeypatch.setattr(module, 'load_available_datasets', lambda train=True: {module.args.dataset_name: data_loader})
        server = module.create_server()
    manager = SimpleNamespace(num_available=lambda: 2,
        sample=lambda num_clients, min_num_clients: [SimpleNamespace(cid=str(i)) for i in range(num_clients)])
    instructions = server.strategy.configure_fit(1, server.strategy.initial_parameters, manager)
    assert len(instructions) == 1
