# Task 4 — submission 9777068

Submitted image: `task4.sif`.

This is the run-21791 full fine-tune of `dolphins_xlstm_unet_b`: T2 input,
three output classes, `128 x 96 x 96` foreground-aware patches, generalized
Dice/focal loss, LR `2e-4`, 200 epochs, and deep supervision. The checkpoint
maximizes mean foreground Dice.

```bash
export PRETRAIN_CHECKPOINT=/path/to/brainaxl_run19726.ckpt
export ASPARAGUS_DATA=/path/to/processed/data
bash train.sh
```

The submitted inference uses Gaussian sliding windows with 0.625 overlap,
eight flip combinations, threshold 0.30, minimum component size 4, at most two
components per class, and hole filling.

```bash
export FOMO26_TASK4_CHECKPOINT=/path/to/best.ckpt
python predict.py --t2 t2.nii.gz --output segmentation.nii.gz
```
