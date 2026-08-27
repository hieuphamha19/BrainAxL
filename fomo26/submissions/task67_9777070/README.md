# Tasks 6 and 7 — submission 9777070

Submitted image: `task6.sif`, used for both linear probing and bias/fairness.

There is deliberately no label fine-tuning in this submission. The run-19726
BrainAxL encoder and its semantic projector are frozen. The code extracts
overlapping `64^3` patches with stride 32, pools mean+max over the five encoder
stages, applies the 512-dimensional semantic projector, then mean+max pools
across patches to produce a 1,024-dimensional vector. The challenge evaluator
fits the downstream linear probe on these vectors.

```bash
export FOMO26_EMBED_CHECKPOINT=/path/to/brainaxl_run19726.ckpt
python predict.py --input image.nii.gz --output embedding.npy
```

The `predict.py` here was recovered from `task6.sif`. It is the semantic-projector
64/32 recipe used in submission 9777070, not the later encoder-pooled 96/32
development variant.
