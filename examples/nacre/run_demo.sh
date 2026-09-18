#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python -m guide --config "$ROOT/configs/nacre_demo.yaml" --method guide "$@"
