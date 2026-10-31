# Reproduce BrainAxL from the main code backup

This repository backs up the reported BrainAxL run and selected FOMO26 TEST
recipes. The main code includes architecture, objectives, data preparation,
training, feature extraction, head fitting, and inference.

## Entry points

| Stage | Code / recipe |
| --- | --- |
| BrainAxL architecture | `asparagus/asparagus/modules/networks/brainaxl.py` |
| Historical checkpoint architecture | `asparagus/asparagus/modules/networks/dolphins_xlstm_unet.py` |
| Pretraining objective | `asparagus/asparagus/modules/lightning_modules/self_supervised.py` |
| NIfTI-backed pretraining task | `scripts/create_fomo300k_nifti_backed_task.py` |
| Coverage-weighted training split | `scripts/create_coverage_weighted_task.py` |
| Reported pretraining recipe | `asparagus/configs/projects/brainaxl/pretrain.yaml` |
| Generic fine-tuning | `asparagus/configs/projects/brainaxl/finetune_{cls,reg,seg}.yaml` |
| Exact Task 1 adaptation | `fomo26/submissions/task1_9777066/{extract_features,fit_head}.py` |
| Exact Task 2 fine-tuning | `fomo26/submissions/task2_9777071/train_folds.sh` |
| Exact Task 3 fine-tuning | `fomo26/submissions/task3_9777067/train_folds.sh` |
| Exact Task 4 fine-tuning | `fomo26/submissions/task4_9777068/train.sh` |
| Exact Task 5 adaptation | `fomo26/submissions/task5_9777069/{extract_features,fit_head}.py` |
| Task 6/7 frozen representation | `fomo26/submissions/task67_9777070/predict.py` |
| Submitted inference and checksums | `fomo26/submissions/` |

The historical model configs `model/dolphins_xlstm_unet_b.yaml` and
`model/core/dolphins_xlstm_unet.yaml` are included because the exact Task 2/4
training scripts select `+model=dolphins_xlstm_unet_b`. They are dependencies
of the selected recipes, rather than additional experiment variants.

## Prepare the reported pretraining task

Install the packages and configure the environment as described in the
[root README](../README.md#installation). Obtain the authorized FOMO300K
NIfTI volumes and original manifest CSV separately. The manifest must have a
`path` column; relative entries are resolved against `FOMO300K_ROOT`.
Run from the repository root:

```bash
export FOMO300K_MANIFEST=/path/to/original/fomo300k_train.csv
export FOMO300K_ROOT=/path/to/FOMO26
export ASPARAGUS_DATA=/path/to/processed/tasks

python scripts/create_fomo300k_nifti_backed_task.py \
  --manifest-csv "$FOMO300K_MANIFEST" \
  --data-root "$FOMO300K_ROOT" \
  --asparagus-data-root "$ASPARAGUS_DATA" \
  --task-name PT902_FOMO300K_FULL \
  --split-name split_95_5_0 \
  --path-column path --val-fraction 0.05 --seed 42 --check-paths

python scripts/create_coverage_weighted_task.py \
  --source-task-dir "$ASPARAGUS_DATA/PT902_FOMO300K_FULL" \
  --output-data-root "$ASPARAGUS_DATA" \
  --task-name PT903_FOMO300K_FULL_COV64_R2 \
  --split-name split_95_5_0 \
  --patch-size 64 64 64 --reference-grid-patches 24 \
  --min-repeat 1 --max-repeat 2 --val-repeat 1 --num-workers 8

cd asparagus
asp_pretrain --config-name projects/brainaxl/pretrain
```

The first script preserves the original manifest order before a seeded
shuffle. Use the original manifest order and contents to recover the same
95/5 split; a seed alone does not reconstruct it from a reordered manifest.
Volumes retain native spacing/orientation at this stage; normalization and
augmentation occur in the pretraining pipeline.

Coverage weighting repeats each training scan according to
`clamp(ceil(prod(ceil(shape_i / 64)) / 24), 1, 2)` and leaves validation
unweighted. The original build summary recorded 306,207 unique scans:
290,897 training and 15,310 validation. Coverage weighting yielded 443,479
training entries (138,315 scans once, 152,582 twice), with no shape-reading
errors. These aggregate counts are useful checks; they do not establish
identity of the underlying scans or splits.

## Reproduce the downstream recipes

Follow the [training guide](training.md) and the
[submission-specific instructions](../fomo26/submissions/README.md). Use the
published [foundation checkpoint](model-weights.md), or retrain it above.
The downstream preprocessing modules are backed up under
`asparagus_preprocessing/asparagus_preprocessing/`.

Original data manifests, labels, and fold assignments remain external inputs.
In particular, Task 3 expects `split_probe24842_5fold_train80val20.json` and
`TEST_probe24842_5fold.json` in its processed task directory. Preserve those
original files in the authorized data store for the same fold assignment;
this code backup does not reconstruct private subject assignments from the
public aggregate logs. Task 1/5 feature manifests follow their task READMEs.

## Verify a checkout

```bash
python scripts/verify_release.py
PYTHONPATH=asparagus python -m pytest asparagus/tests/test_brainaxl.py
```

The first command verifies protected submitted sources, syntax, training-log
records, and documentation links. The model tests check architecture and
checkpoint-key compatibility. Re-running training requires the external data,
folds, and compute described above. Optimizer state is not included in the
public weight-only checkpoint, so resuming an interrupted original run also
requires its full training checkpoint.

Exploratory sweeps, discarded architectures, generated outputs, raw data,
credentials, downstream binary weights, and SIF images are outside this main
code backup. The [artifact boundaries](reproducibility.md) explain these
external inputs and the immutable submitted-source policy.
