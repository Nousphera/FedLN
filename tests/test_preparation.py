"""Preparation contracts tested with bounded files, never a corpus download."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture(path, shape=(32,32,3), classes=10):
    rng = np.random.default_rng(7)
    np.savez(path, train_images=rng.integers(0,256,(40,*shape),dtype=np.uint8),
             train_labels=np.arange(40) % classes,
             test_images=rng.integers(0,256,(10,*shape),dtype=np.uint8), test_labels=np.arange(10)%classes)


def test_prepare_import_validate_no_overwrite_and_tamper(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    args = ["--dataset", "cifar10", "--root", str(target)]
    first = prep.main([*args, "--source", str(source)])
    assert first["metadata"]["train_rows"] == 40
    assert prep.main([*args, "--validate"]) == first
    with pytest.raises(FileExistsError):
        prep.main([*args, "--source", str(source)])
    with (target / "cifar10.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        prep.main([*args, "--validate"])


@pytest.mark.parametrize("shape,classes", [((28,28,3),10),((32,32,3),11)])
def test_reject_wrong_shape_or_out_of_range_labels(tmp_path, shape, classes):
    source = tmp_path / "invalid.npz"
    fixture(source, shape, classes)
    with pytest.raises(ValueError):
        prep.main(["--dataset","cifar10","--source",str(source),"--root",str(tmp_path/"target")])
    assert not (tmp_path/"target/cifar10.npz").exists()


def test_prepared_arrays_reach_training_loader(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    root = tmp_path / "data"
    prep.main(["--dataset","cifar10","--source",str(source),"--root",str(root)])
    from data.vision import load_vision
    ds, classes, count, labels, ids = load_vision("cifar10", root=root, num_shards=2, shard_id=0, noise_lvl=None, batch_size=4)
    x, y = next(iter(ds))
    assert classes == 10 and count > 0 and y.shape[1] == 10


def test_public_archive_extraction_blocks_traversal(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as out:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        out.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(archive) as source:
        with pytest.raises(tarfile.FilterError):
            source.extractall(tmp_path / "safe")
    assert not (tmp_path / "escaped").exists()


def test_existing_manifest_cannot_leave_a_half_published_array(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    target.mkdir()
    saved = target / "cifar10.manifest.json"
    saved.write_text('{"existing":true}')
    with pytest.raises((ValueError, FileExistsError)):
        prep.main(["--dataset", "cifar10", "--source", str(source), "--root", str(target)])
    assert not (target / "cifar10.npz").exists()
    assert saved.read_text() == '{"existing":true}'


@pytest.mark.parametrize('prior_root', [None, 'preserved-existing-root'])
def test_audio_import_survives_relocation_and_restores_environment(tmp_path, monkeypatch, prior_root):
    import json
    import os
    from test_audio_data import write_records
    from data.spcm import load_spcm
    source = tmp_path / 'source'
    source.mkdir()
    write_records(source, 'train', 0, 1, [0, 1])
    write_records(source, 'test', 0, 1, [0, 1])
    (source / 'unrelated-private.txt').write_text('not dataset input')
    if prior_root is None:
        monkeypatch.delenv('TFDS_DATA_DIR', raising=False)
    else:
        monkeypatch.setenv('TFDS_DATA_DIR', prior_root)
    target = tmp_path / 'imported'
    args = ['--dataset', 'spcm', '--root', str(target)]
    result = prep.main([*args, '--source', str(source)])
    assert os.environ.get('TFDS_DATA_DIR') == prior_root
    assert prep.main([*args, '--validate']) == result
    assert os.environ.get('TFDS_DATA_DIR') == prior_root
    assert not (target / 'unrelated-private.txt').exists()
    assert '.prepare-' not in json.dumps(result) and str(source) not in json.dumps(result)
    assert not list(tmp_path.glob('.prepare-*'))
    monkeypatch.setenv('TFDS_DATA_DIR', str(target))
    data, classes, count, _, _ = load_spcm(0, 1, batch_size=2, shuffle=False)
    assert classes == 12 and count == 2
    assert next(iter(data))[0].shape == (2, 96, 64, 1)


def test_array_publication_failure_rolls_back_both_files(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    real_link = prep.os.link
    calls = []
    def fail_manifest(source, destination):
        calls.append(destination)
        if str(destination).endswith(".manifest.json"):
            raise OSError("publication failed")
        return real_link(source, destination)
    monkeypatch.setattr(prep.os, "link", fail_manifest)
    with pytest.raises(OSError, match="publication failed"):
        prep.main(["--dataset", "cifar10", "--source", str(source), "--root", str(target)])
    assert len(calls) == 2
    assert not (target / "cifar10.npz").exists()
    assert not (target / "cifar10.manifest.json").exists()
    assert not list(target.glob(".prepare-*"))
