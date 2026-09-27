#!/usr/bin/env bash
# Reproduce the paper's metric benchmark from the released data.
#
#   scripts/reproduce.sh            score + evaluate using the released statements
#   scripts/reproduce.sh --full     also retrain the statement model and regenerate statements
#
# Extra arguments after the flag are passed to every cap-eval call, e.g.
#   scripts/reproduce.sh --set device=cuda
set -euo pipefail
cd "$(dirname "$0")/.."

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
    shift
fi

INPUT=data/CAP-Correctness.csv
if [[ $FULL == 1 ]]; then
    cap-eval "$@" train-statements
    cap-eval "$@" generate-statements --model-dir outputs/statement_model/final
    INPUT=outputs/statements.csv
fi

cap-eval "$@" score --input "$INPUT"
cap-eval "$@" evaluate --input "$INPUT"
