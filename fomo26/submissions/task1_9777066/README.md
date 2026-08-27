# Task 1 — submission 9777066

Submitted image: `task1_ens.sif`.

This is a frozen BrainAxL probe. FLAIR, ADC, and DWI b=1000 are independently
foreground-normalized, cropped, resized to `128 x 128 x 32`, and encoded. The
head averages standardized logits from all-stage mean+max features (7,440-D)
and stage-3 mean+max features (960-D). Both balanced logistic heads use
`C=0.05` and are fitted on all 21 fine-tuning subjects.

```bash
export FOMO26_T1_CHECKPOINT=/path/to/brainaxl_run19726.ckpt
python extract_features.py --manifest train.csv --output features.npz
python fit_head.py --features features.npz \
  --output models/task1_head_params_3ch_ens_mm_s3.npz

export FOMO26_T1_HEAD=$PWD/models/task1_head_params_3ch_ens_mm_s3.npz
python predict.py --flair flair.nii.gz --adc adc.nii.gz \
  --dwi dwi_b1000.nii.gz --output prediction.txt
```

`train.csv` columns are `subject_id,flair,adc,dwi_b1000,label`. The output head
and run-19726 checkpoint occupy `/app/models/` in the submitted image.
