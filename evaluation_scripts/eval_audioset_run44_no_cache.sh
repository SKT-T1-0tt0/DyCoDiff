#!/usr/bin/env bash
# Evaluation command used for the AudioSet-Drums-VAT run-44 comparison.
# Run from the repository root after activating the evaluation environment:
#   conda activate eval_mcfl

set -euo pipefail
cd "$(dirname "$0")/.."

python3 eval_all.py \
  --real_dir results/9_tacm_/real \
  --baseline_dir results/9_tacm_/fake1_30fps \
  --mcfl_dir results/44_tacm_audioset_drums/fake1_30fps \
  --metrics fvd fid ffc clip av_align tc_flicker \
  --output evaluation_report_44_tacm_audioset.txt \
  --no_cache
