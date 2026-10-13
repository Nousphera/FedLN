#!/usr/bin/env bash
# Nousphera maintenance: reliable dispatch and dependency-free launcher help.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: ./src/run.sh METHOD [experiment arguments]' \
        '' \
        'Methods:' \
        '  nnc                  Nearest Neighbor Correction' \
        '  na_fedavg            Noise-Aware Federated Averaging' \
        '  akd                  Adaptive Knowledge Distillation' \
        '  fedavg               Federated Averaging' \
        '  bi_tempered_loss      FedAvg with bi-tempered logistic loss' \
        '  label_smoothing      FedAvg with label smoothing' \
        '  confidence_learning  FedAvg with confident learning' \
        '  centralized          Centralized training' \
        '' \
        'Use METHOD --help to list experiment arguments.' \
        'Relative data/cache paths are resolved from src/. Activate your Python environment first.'
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; exit 2 ;;
    centralized) entry="baselines/centralized.py" ;;
    fedavg) entry="baselines/fedavg__standard.py" ;;
    confidence_learning) entry="baselines/fedavg__cl.py" ;;
    bi_tempered_loss) entry="baselines/fedavg__loss.py" ;;
    label_smoothing) entry="baselines/fedavg__smooth.py" ;;
    nnc) entry="fedln/fedln__nnc.py" ;;
    na_fedavg) entry="fedln/fedln__nafedavg.py" ;;
    akd) entry="fedln/fedln__akd.py" ;;
    *) printf 'Unknown method: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac
shift
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python "$entry" "$@"
