# Reproducibility guide

BrainAxL separates the reusable framework, experiment configuration, immutable
submission payloads, and large research artifacts. This keeps the repository
auditable without redistributing restricted data or multi-gigabyte weights.

## Reproduction levels

| Level | Public content | External requirement |
|---|---|---|
| Architecture | `brainaxl.py` and unit tests | Python environment |
| Pretraining recipe | BrainAxL Hydra config and method specification | Processed FOMO300K data |
| Generic fine-tuning | Classification, regression, and segmentation configs | Processed downstream data and pretrained checkpoint |
| FOMO26 TEST submissions | Versioned training/adaptation code, exact SIF inference source, manifests | Challenge data and recorded checkpoints |
| Training evidence | Sanitized epoch CSVs and selected-checkpoint summaries | None for inspection; data/checkpoints for reruns |

The repository does not contain MRI data, subject metadata, checkpoints, SIF
images, cached embeddings, or predictions.

The public [`training_logs/`](../training_logs/README.md) directory records the
reported pretraining history, full-fine-tuning histories for Tasks 3 and 4,
selected fold manifests for Task 2, and fit summaries for frozen-probe Tasks 1
and 5. These are local-validation records, not hidden TEST metrics.

## Canonical configuration

The reported pretraining configuration is
`asparagus/configs/projects/brainaxl/pretrain.yaml`. It references task
`PT903_FOMO300K_FULL_COV64_R2` and produces a Lightning checkpoint compatible
with both `brainaxl_b` and the historical `dolphins_xlstm_unet_b` name.

Generic downstream configs live beside it:

- `finetune_cls.yaml`
- `finetune_reg.yaml`
- `finetune_seg.yaml`

These generic configs demonstrate the public framework. For the exact FOMO26
TEST artifacts, use `fomo26/submissions/` instead; each directory is keyed by
the challenge submission ID.

## Submission provenance

`fomo26/submissions/artifact_manifest.json` maps submission ID, task, SIF name,
SIF SHA-256, embedded source path, and embedded-source SHA-256. Files listed in
`sources.sha256` are immutable historical payloads extracted from the validated
images.

Run the complete dependency-free integrity gate from the repository root:

```bash
python scripts/verify_release.py
```

The gate verifies:

- all six expected submission IDs and their SIF/source mapping;
- cryptographic hashes of protected inference sources;
- absence of checkpoints, SIFs, NIfTI files, arrays, and feature caches;
- Python, shell, and JSON syntax;
- local Markdown links and canonical repository identity.

## Checkpoint contract

All six submissions initialize from the same run-19726 BrainAxL SSL checkpoint.
Task-specific directories document the expected runtime filenames and
environment variables. Checkpoints should be transferred through an approved
artifact store and verified independently; they should never be committed to
Git.

Published state dictionaries retain historical key names. The public model
keeps those names intentionally, and `asparagus/tests/test_brainaxl.py` guards
shape-level compatibility.

Selected pretrained and downstream state dictionaries are consolidated in one
[Hugging Face model repository](model-weights.md). The GitHub repository stores
their layout and provenance but deliberately excludes the binary weights.

## Adding a new artifact version

Do not overwrite a protected historical payload. Create a new directory whose
name identifies the task and artifact version, add a new manifest record, and
record hashes after the artifact is finalized. Describe any difference between
training code, packaged inference, and later refactoring explicitly.

## Determinism and limitations

The release records seeds and selected hyperparameters, but bitwise training
reproduction can still depend on CUDA, cuDNN, PyTorch, data ordering, and image
preprocessing versions. Challenge TEST data and official evaluator behavior are
external to this repository. The code is a research artifact, not a clinically
validated system.
