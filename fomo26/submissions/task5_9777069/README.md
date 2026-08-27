# Task 5 — submission 9777069

Submitted image: `task5_deconf_ap140_v3.sif`.

This is a frozen run-19726 probe, not an end-to-end fine-tune. T1 is sampled on
a world-RAS `200 x 140 x 230 mm` box at 2 mm spacing, centered on the largest
nonzero foreground component, and constant-padded to `128^3`. Only the
320-dimensional mean-pooled stage-3 feature is used.

The balanced logistic head (`C=0.05`) is fitted on both the original and
pydeface-2.0.2 version of each of the 48 fine-tuning subjects. This matches the
defaced validation/TEST domain while retaining the original-domain examples.

```bash
export FOMO26_T5_CHECKPOINT=/path/to/brainaxl_run19726.ckpt
python extract_features.py --manifest train_domains.csv \
  --output features.npz
python fit_head.py --features features.npz \
  --output models/task5_head_params_t1_ap140_s3mean_domainaug_C0.05.npz

export FOMO26_T5_HEAD=$PWD/models/task5_head_params_t1_ap140_s3mean_domainaug_C0.05.npz
python predict.py --t1 t1.nii.gz --output probability.txt
```

`train_domains.csv` columns are `subject_id,t1,label,domain`; every subject must
have one `original` row and one `defaced` row. The feature extractor imports the
submitted inference preprocessing so training/inference geometry cannot drift.
