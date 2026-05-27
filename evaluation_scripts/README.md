# Evaluation Scripts

This folder stores concrete evaluation commands used during DyCoDiff experiments. The scripts assume that generated videos and ground-truth videos are already placed under `results/`.

## Environment

Use the evaluation environment exported in `environment_eval.yml`:

```bash
conda activate eval_mcfl
```

## AudioSet-Drums-VAT Run 44

`eval_audioset_run44_no_cache.sh` evaluates the run-44 generated videos against the AudioSet-Drums-VAT real videos and baseline videos:

```bash
bash evaluation_scripts/eval_audioset_run44_no_cache.sh
```

It is equivalent to:

```bash
python3 eval_all.py \
  --real_dir results/9_tacm_/real \
  --baseline_dir results/9_tacm_/fake1_30fps \
  --mcfl_dir results/44_tacm_audioset_drums/fake1_30fps \
  --metrics fvd fid ffc clip av_align tc_flicker \
  --output evaluation_report_44_tacm_audioset.txt \
  --no_cache
```

The output report is ignored by Git through `evaluation_report*.txt/json` rules.
