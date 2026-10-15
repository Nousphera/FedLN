"""Shared experiment configuration."""

import math

DATASETS = ("cifar10", "fashionmnist", "pathmnist", "eurosat", "spcm", "cifar100", "cifar10n", "cifar100n")
METHODS = {
    "nnc": "Nearest-neighbor correction", "akd": "Adaptive knowledge distillation",
    "nafedavg": "Noise-aware federated averaging", "fedavg": "Federated averaging",
    "label-smoothing": "Label smoothing", "bi-tempered": "Bi-tempered logistic loss",
    "confident-learning": "Confident learning", "centralized": "Centralized training",
}


def defaults():
    return {
        "method": "nnc", "dataset": "cifar10", "model": "auto", "device": "cpu",
        "num-clients": 30, "num-rounds": 200, "participation": .8,
        "quantity-variation": .25, "local-epochs": 1, "batch-size": 128,
        "optimizer": "auto", "learning-rate": 0., "momentum": .9,
        "noisy-clients": .8, "noise-level": .4, "noise-sparsity": .7,
        "neighbors": 100, "noise-threshold": .15, "noise-round": 25,
        "percentile": 75., "teacher": "embeddings", "kd-beta": 10., "temperature": 2.,
        "label-smoothing": .2, "t1": .8, "t2": 1.4,
        "cleanlab-folds": 5, "cleanlab-epochs": 20,
        "seed": 42, "torch-threads": 1, "num-workers": 0,
        "data-root": "data", "feature-root": "features", "human-noise-key": "", "output": "results/fedln.json",
        "registration-timeout": 120., "reply-timeout": 86400.,
    }


def validate(cfg):
    cfg = dict(cfg)
    for key, value in cfg.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if cfg["dataset"] not in DATASETS or cfg["method"] not in METHODS:
        raise ValueError("Unknown dataset or method")
    if cfg["model"] not in {"auto", "resnet20", "cnn", "convmixer"}:
        raise ValueError("Unknown model")
    audio = cfg["dataset"] == "spcm"
    if (audio and cfg["model"] not in {"auto", "cnn"}) or (not audio and cfg["model"] == "cnn"):
        raise ValueError("CNN is the audio model; ResNet20 and ConvMixer are vision models")
    for key in ("num-clients", "num-rounds", "local-epochs", "batch-size", "neighbors", "noise-round",
                "cleanlab-folds", "cleanlab-epochs", "torch-threads"):
        if type(cfg[key]) != int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["cleanlab-folds"] < 2:
        raise ValueError("Confident learning requires at least two folds")
    for key in ("num-workers", "seed"):
        if type(cfg[key]) != int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("quantity-variation", "noisy-clients", "noise-level", "noise-sparsity", "noise-threshold", "momentum"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not 0 < cfg["participation"] <= 1 or not 0 < cfg["percentile"] < 100:
        raise ValueError("Invalid participation or percentile")
    if cfg["method"] == "nafedavg" and cfg["noise-round"] > cfg["num-rounds"]:
        raise ValueError("noise-round must not exceed rounds for NA-FedAvg")
    if cfg["teacher"] not in {"embeddings", "global-logits"}:
        raise ValueError("Invalid AKD teacher")
    if cfg["human-noise-key"] and cfg["dataset"] not in {"cifar10n", "cifar100n"}:
        raise ValueError("human-noise-key applies only to CIFAR-N")
    if cfg["kd-beta"] < 0 or cfg["temperature"] <= 0:
        raise ValueError("Invalid distillation parameters")
    if not 0 <= cfg["label-smoothing"] < 1 or not 0 < cfg["t1"] <= 1 or not 1 <= cfg["t2"] < 2:
        raise ValueError("Invalid robust loss parameters")
    if cfg["optimizer"] == "auto":
        cfg["optimizer"] = "sgd" if cfg["model"] != "convmixer" and (cfg["dataset"].startswith("cifar") or cfg["dataset"] == "fashionmnist") else "adam"
    if cfg["optimizer"] not in {"sgd", "adam"} or cfg["learning-rate"] < 0:
        raise ValueError("Invalid optimizer or learning rate")
    if cfg["learning-rate"] == 0:
        cfg["learning-rate"] = .1 if cfg["optimizer"] == "sgd" else .001
    if cfg["registration-timeout"] <= 0 or cfg["reply-timeout"] <= 0:
        raise ValueError("Timeouts must be positive")
    if not (cfg["device"] in {"cpu", "mps", "cuda"} or cfg["device"].startswith("cuda:")):
        raise ValueError("device must be cpu, mps or cuda")
    return cfg
