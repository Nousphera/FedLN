"""FedLN supervision and the bi-tempered logistic baseline."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _log_t(x, temperature):
    return x.log() if temperature == 1. else (x.pow(1. - temperature) - 1.) / (1. - temperature)


def _exp_t(x, temperature):
    if temperature == 1.:
        return x.exp()
    return (1. + (1. - temperature) * x).clamp_min(0.).pow(1. / (1. - temperature))


class _TemperedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, temperature, num_iters):
        centered = logits - logits.amax(dim=-1, keepdim=True)
        if temperature > 1.:
            scaled = centered
            for _ in range(num_iters):
                partition = _exp_t(scaled, temperature).sum(dim=-1, keepdim=True)
                scaled = centered * partition.pow(1. - temperature)
            partition = _exp_t(scaled, temperature).sum(dim=-1, keepdim=True)
            normalization = -_log_t(partition.reciprocal(), temperature)
        else:
            lower = torch.zeros_like(centered[..., :1])
            upper = lower - _log_t(logits.new_tensor(1. / logits.shape[-1]), temperature)
            for _ in range(num_iters):
                midpoint = (lower + upper) / 2.
                partition = _exp_t(centered - midpoint, temperature).sum(dim=-1, keepdim=True)
                lower = torch.where(partition > 1., midpoint, lower)
                upper = torch.where(partition > 1., upper, midpoint)
            normalization = (lower + upper) / 2.
        probabilities = _exp_t(centered - normalization, temperature)
        ctx.save_for_backward(probabilities)
        ctx.temperature = temperature
        return probabilities

    @staticmethod
    def backward(ctx, gradient):
        probabilities, = ctx.saved_tensors
        power = probabilities.pow(ctx.temperature)
        scaled = gradient * power
        return scaled - power * scaled.sum(dim=-1, keepdim=True) / power.sum(dim=-1, keepdim=True), None, None


def tempered_softmax(logits: Tensor, temperature: float, num_iters: int = 50) -> Tensor:
    if temperature <= 0 or num_iters < 1:
        raise ValueError("Temperature and normalization iterations must be positive")
    if temperature == 1.:
        return logits.softmax(dim=-1)
    return _TemperedSoftmax.apply(logits, float(temperature), int(num_iters))


def bi_tempered_loss(logits: Tensor, labels: Tensor, t1: float = .8, t2: float = 1.4,
                     label_smoothing: float = 0., num_iters: int = 50,
                     reduction: str = "mean") -> Tensor:
    """Bi-tempered Bregman loss; t1=t2=1 recovers cross entropy."""
    if not 0 < t1 < 2 or not 0 <= label_smoothing < 1:
        raise ValueError("Require 0 < t1 < 2 and 0 <= label_smoothing < 1")
    target = F.one_hot(labels.long(), logits.shape[-1]).to(logits.dtype) if labels.ndim == logits.ndim - 1 else labels.to(logits.dtype)
    if target.shape != logits.shape:
        raise ValueError("Labels must be class indices or distributions matching logits")
    if label_smoothing:
        classes = logits.shape[-1]
        target = (1. - classes / (classes - 1.) * label_smoothing) * target + label_smoothing / (classes - 1.)
    probabilities = tempered_softmax(logits, t2, num_iters)
    tiny = torch.finfo(logits.dtype).tiny
    log_probabilities = F.log_softmax(logits, dim=-1) if t1 == t2 == 1. else _log_t(probabilities.clamp_min(tiny), t1)
    logarithmic = target * (_log_t(target.clamp_min(tiny), t1) - log_probabilities)
    correction = (target.pow(2. - t1) - probabilities.pow(2. - t1)) / (2. - t1)
    values = (logarithmic - correction).sum(dim=-1)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction != "mean":
        raise ValueError("Reduction must be mean, sum or none")
    return values.mean()


class BiTemperedLogisticLoss(nn.Module):
    def __init__(self, t1=.8, t2=1.4, label_smoothing=0., num_iters=50):
        super().__init__()
        self.t1, self.t2 = t1, t2
        self.label_smoothing, self.num_iters = label_smoothing, num_iters

    def forward(self, logits, labels):
        return bi_tempered_loss(logits, labels, self.t1, self.t2,
                                self.label_smoothing, self.num_iters)


def forward_kl(student: Tensor, teacher: Tensor, temperature: float = 2.) -> Tensor:
    """Equation 9: T² KL(student || frozen round-global teacher)."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    student_log = F.log_softmax(student / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher.detach() / temperature, dim=-1)
    return temperature ** 2 * (student_log.exp() * (student_log - teacher_log)).sum(dim=-1).mean()
