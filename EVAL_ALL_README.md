# Evaluation Guide

DyCoDiff reports six metrics: FVD, FID, FFC, CLIP, AV-align, and TC_FLICKER.

## Environment

The evaluation environment used for metric computation is exported as `environment_eval.yml` from the `eval_mcfl` conda environment:

```bash
conda env create -f environment_eval.yml
conda activate eval_mcfl
```

## Paper-Style Evaluation

Use `eval_all_three_groups.py` to reproduce the three-dataset comparison layout used in the paper:

```bash
python eval_all_three_groups.py \
  --metrics fvd fid ffc clip av_align tc_flicker \
  --output evaluation_report_three_groups.txt
```

Default run mapping:

| Dataset | Baseline | Only-MCFL | DyCoDiff |
| --- | ---: | ---: | ---: |
| URMP-VAT | 3 | 6 | 12 |
| Landscape-VAT | 7 | 8 | 13 |
| AudioSet-Drums-VAT | 9 | 10 | 11 |

The script expects generated videos under `results/{run}_tacm_/fake1_30fps/`, ground-truth videos under `results/{run}_tacm_/real/`, and audio files under `results/{run}_tacm_/audio/`.

## Custom Evaluation

Use `eval_all.py` for a custom comparison:

```bash
python eval_all.py \
  --real_dir results/9_tacm_/real \
  --baseline_dir results/9_tacm_/fake1_30fps \
  --mcfl_dir results/10_tacm_/fake1_30fps \
  --mcfl2_dir results/11_tacm_/fake1_30fps \
  --metrics fvd fid ffc clip av_align tc_flicker \
  --output evaluation_report_audioset.txt
```

Concrete commands used in experiments are stored under `evaluation_scripts/`. For example, `evaluation_scripts/eval_audioset_run44_no_cache.sh` records the AudioSet-Drums-VAT run-44 no-cache evaluation command.

## Metrics

| Metric | Direction | Purpose | Main dependency |
| --- | --- | --- | --- |
| FVD | lower is better | video distribution quality | I3D features |
| FID | lower is better | frame-level visual quality | Inception features |
| FFC | higher is better in the paper tables | first-frame/reference-image consistency | CLIP-style visual similarity or project-specific implementation |
| CLIP | higher is better | text-video semantic similarity | OpenAI CLIP and `prompts.txt` |
| AV-align | higher is better | audio-video correspondence | motion/audio energy correlation |
| TC_FLICKER | lower is better | temporal flicker | frame-difference statistics |

Note: this repository also contains an optical-flow based `eval_ffc.py` inherited from earlier experiments. If you need to exactly reproduce the paper's First-frame Consistency values, verify that the selected FFC implementation matches the paper definition before reporting numbers.

## Caches and Weights

Evaluation may create local caches such as `fvd_cache/`, `fid_cache/`, and `ffc_cache/`; these are ignored by Git. RAFT, I3D, CLIP, and other evaluation weights should be downloaded locally and must not be committed.
