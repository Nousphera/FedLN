"""Shared PyTorch optimization and evaluation for every experiment."""

import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from .models import build_model, regularization_loss
from .losses import bi_tempered_loss, forward_kl


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LocalData(Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset, self.indices = dataset, np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.indices) != len(self.labels) or not len(self.indices):
            raise ValueError("Local data must contain aligned examples and labels")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = int(self.indices[row])
        x, _, _ = self.dataset[index]
        return x, int(self.labels[row]), index


def loader(dataset, cfg, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=cfg["batch-size"], shuffle=shuffle,
                      num_workers=cfg["num-workers"], generator=torch.Generator().manual_seed(seed),
                      pin_memory=cfg["device"].startswith("cuda"))


def make_optimizer(parameters, cfg):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, lr=cfg["learning-rate"], momentum=cfg["momentum"])
    return torch.optim.Adam(parameters, lr=cfg["learning-rate"], eps=1e-7)


def train_local(model, data, cfg, seed, *, features=None, projection=None, teacher=None,
                optimizer_state=None, epochs=None):
    seed_everything(seed, cfg["torch-threads"])
    device = torch.device(cfg["device"])
    model.to(device).train()
    parameters = list(model.parameters())
    if projection is not None:
        projection.to(device).train()
        parameters += list(projection.parameters())
    if teacher is not None:
        teacher.to(device).eval().requires_grad_(False)
    optimizer = make_optimizer(parameters, cfg)
    if optimizer_state:
        for key, value in optimizer_state.items():
            index, name = key.split("/", 1)
            parameter = parameters[int(index)]
            optimizer.state[parameter][name] = value.to(device) if name != "step" else value.cpu()
    loss_sum, correct, total = 0., 0, 0
    for epoch in range(epochs or cfg["local-epochs"]):
        for x, y, rows in loader(data, cfg, shuffle=True, seed=seed + epoch):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if projection is not None:
                logits, pooled = model(x, return_features=True)
            else:
                logits = model(x)
            if cfg["method"] == "bi-tempered":
                objective = bi_tempered_loss(logits, y, t1=cfg["t1"], t2=cfg["t2"])
            else:
                smoothing = cfg["label-smoothing"] if cfg["method"] == "label-smoothing" else 0.
                objective = F.cross_entropy(logits, y, label_smoothing=smoothing)
            if projection is not None:
                target = torch.from_numpy(np.asarray(features[rows.numpy()]).copy()).to(device=device, dtype=logits.dtype)
                objective = objective + cfg["kd-beta"] * F.l1_loss(projection(pooled), target)
            elif teacher is not None:
                with torch.no_grad():
                    target = teacher(x)
                objective = objective + cfg["kd-beta"] * forward_kl(logits, target, cfg["temperature"])
            objective = objective + regularization_loss(model)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training loss")
            objective.backward()
            optimizer.step()
            loss_sum += objective.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
    saved_optimizer = {f"{i}/{name}": value.detach().cpu().clone()
                       for i, parameter in enumerate(parameters)
                       for name, value in optimizer.state.get(parameter, {}).items()}
    model.cpu()
    if projection is not None:
        projection.cpu()
    return {"loss": loss_sum / total, "accuracy": correct / total}, saved_optimizer


class _AudioSegments(Dataset):
    def __init__(self, data):
        self.data = data
        self.source = data.dataset if isinstance(data, LocalData) else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, row):
        from .audio import audio_spectrogram
        index = int(self.data.indices[row]) if isinstance(self.data, LocalData) else row
        waveform = self.source.raw(index)
        segments = torch.stack([audio_spectrogram(waveform[start:start + 16000])
                                for start in range(0, len(waveform), 16000)])
        return segments, int(self.data.labels[row])


def _collate_audio(rows):
    segments, labels = zip(*rows)
    return torch.cat(segments), [len(clip) for clip in segments], torch.tensor(labels)


@torch.inference_mode()
def predict(model, data, cfg, *, energies=False):
    device = torch.device(cfg["device"])
    model.to(device).eval()
    outputs, labels = [], []
    if cfg["dataset"] == "spcm":
        batches = DataLoader(_AudioSegments(data), batch_size=cfg["batch-size"],
                             num_workers=cfg["num-workers"], collate_fn=_collate_audio,
                             pin_memory=cfg["device"].startswith("cuda"))
        for segments, counts, target in batches:
            logits = torch.cat([model(chunk.to(device)).cpu()
                                for chunk in segments.split(cfg["batch-size"])])
            clip_outputs = []
            for clip in logits.split(counts):
                if energies:
                    result = torch.logsumexp(clip, dim=1).mean()
                elif len(clip) == 1:
                    result = clip[0]
                else:
                    # Average segment probabilities; log values remain valid CE inputs.
                    result = torch.logsumexp(clip.log_softmax(dim=-1), dim=0) - math.log(len(clip))
                clip_outputs.append(result)
            outputs.append(torch.stack(clip_outputs))
            labels.append(target)
    else:
        for x, y, _ in loader(data, cfg):
            logits = model(x.to(device))
            outputs.append((torch.logsumexp(logits, 1) if energies else logits).cpu())
            labels.append(y)
    model.cpu()
    return torch.cat(outputs), torch.cat(labels)


def evaluate(model, data, cfg):
    logits, labels = predict(model, data, cfg)
    loss = float(F.cross_entropy(logits, labels))
    if not np.isfinite(loss):
        raise FloatingPointError("Nonfinite evaluation loss")
    return {"loss": loss, "accuracy": float((logits.argmax(1) == labels).float().mean())}


def confident_keep(dataset, evaluation_dataset, indices, labels, cfg, seed):
    """Find label issues from exclusively out-of-fold local predictions."""
    from cleanlab.filter import find_label_issues
    from sklearn.model_selection import KFold, StratifiedKFold

    indices, labels = np.asarray(indices), np.asarray(labels)
    counts = np.unique(labels, return_counts=True)[1]
    if len(counts) < 1 or len(indices) != len(labels):
        raise ValueError("Confident learning requires aligned nonempty examples and labels")
    if len(counts) == 1:
        # A single observed class cannot identify cross-class label errors.
        return np.arange(len(indices))
    if counts.min() >= 2:
        folds = min(cfg["cleanlab-folds"], int(counts.min()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        # Singleton classes cannot be stratified. Keep a fixed class head and
        # genuinely held-out predictions, even when a training fold lacks a class.
        folds = min(cfg["cleanlab-folds"], len(indices))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    seed_everything(cfg["seed"], cfg["torch-threads"])
    initial = build_model(cfg["dataset"], cfg["model"]).state_dict()
    # Class space is fixed by the architecture, even if a shard lacks a class.
    classes = {"cifar100": 100, "cifar100n": 100, "pathmnist": 9, "spcm": 12}.get(cfg["dataset"], 10)
    probabilities = np.zeros((len(indices), classes), dtype=np.float64)
    fold_cfg = cfg | {"method": "fedavg"}
    for fold, (train, held_out) in enumerate(splitter.split(indices, labels)):
        model = build_model(cfg["dataset"], cfg["model"])
        model.load_state_dict(initial)
        train_local(model, LocalData(dataset, indices[train], labels[train]), fold_cfg,
                    seed + fold, epochs=cfg["cleanlab-epochs"])
        logits, _ = predict(model, LocalData(evaluation_dataset, indices[held_out], labels[held_out]), cfg)
        probabilities[held_out] = logits.softmax(1).numpy()
    issues = find_label_issues(labels=np.asarray(labels), pred_probs=probabilities,
                               filter_by="confident_learning", n_jobs=1)
    keep = np.flatnonzero(~issues)
    if not len(keep):
        raise ValueError("Confident learning removed every local example")
    return keep
