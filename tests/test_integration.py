"""Actual CPU client/server training through stock Flower protobuf messages."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('method', ['fedavg', 'nnc', 'akd', 'akd_global', 'na_fedavg'])
def test_original_methods_train_and_aggregate(tmp_path, method):
    output = tmp_path / method
    command = [sys.executable, str(ROOT/'run.py'), '--method', 'akd' if method=='akd_global' else method, '--output', str(output),
               '--dataset_name', 'synthetic', '--num_clients', '2', '--num_rounds', '2',
               '--train_epochs', '1', '--batch_size', '8', '--seed', '17']
    if method in ['nnc', 'akd', 'akd_global']:
        command += ['--knn_neighbors', '5']
    if method=='akd_global':
        command += ['--akd_teacher','global_logits']
    if method == 'na_fedavg':
        command += ['--est_noise_round', '1', '--participation_rate', '.5']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output/'results.json').read_text())
    assert report['backend'] == 'sequential-stock-flower'
    assert report['rounds'][0]['parameters_changed']
    for r in report['rounds']:
        assert r['parameters_changed'] or r['aggregation_metrics'].get('update_skipped_zero_impact')
    assert report['rounds'][0]['examples'] == 32
    assert (output/'final-parameters.npz').is_file()
    assert not list(output.glob('cache-*'))
    if method in ['akd','akd_global']:
        assert any(m['distillation_active'] for m in report['rounds'][1]['client_metrics'])
    if method == 'na_fedavg':
        assert len(report['rounds'][0]['clients']) == 2
        assert len(report['rounds'][1]['clients']) == 1
