# Task 3 — submission 9777067

Submitted image: `task3_v2_spacing1mm.sif`.

This is a five-fold `dolphins_xlstm_unet_b_clsreg` brain-age ensemble initialized
from run 19726. It uses MSE, encoder LR `1e-4`, head/decoder LR `1e-3`, batch
size 2, 60 epochs, and fold seeds 4610–4614. Training and inference use the same
nonzero crop with 16-voxel margin and `128^3` center crop/pad. Inference first
canonicalizes T1 and resamples anisotropic input to 1 mm isotropic spacing.

```bash
export PRETRAIN_CHECKPOINT=/path/to/brainaxl_run19726.ckpt
export ASPARAGUS_DATA=/path/to/processed/data
bash train_folds.sh
```

Place the selected checkpoints at `model/fold{0..4}_best.ckpt`:

```bash
export FOMO26_TASK3_MODEL_DIR=$PWD/model
python predict.py --t1 t1.nii.gz --output brain_age.txt
```

The submitted checkpoint family is run 25919, not the older whole-volume crop
recipe that appeared in an earlier Task 3 image.
