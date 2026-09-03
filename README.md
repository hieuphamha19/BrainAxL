# BrainAxL

[![CI](https://github.com/hieuphamha19/BrainAxL/actions/workflows/ci.yml/badge.svg)](https://github.com/hieuphamha19/BrainAxL/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Model weights](https://img.shields.io/badge/%F0%9F%A4%97-model_weights-yellow.svg)](https://huggingface.co/hieuphamha/BrainAxL)

Official PyTorch code for **BrainAxL: a multiscale axial LSTM foundation model
for 3D brain MRI**. The repository includes the model, masked-reconstruction +
variance/covariance pretraining objective, and downstream classification,
regression, and segmentation fine-tuning pipelines.

BrainAxL is implemented on top of the open-source
[Asparagus](https://github.com/Sllambias/asparagus) medical-imaging framework
and its [gardening_tools](https://github.com/Sllambias/gardening_tools)
components. The latter is pinned to a public Git commit because it is not
distributed through PyPI.

[Installation](#installation) · [Training guide](docs/training.md) ·
[Training logs](training_logs/README.md) · [Model weights](docs/model-weights.md) ·
[FOMO26 submissions](fomo26/submissions/README.md) ·
[Reproducibility](docs/reproducibility.md) · [Contributing](CONTRIBUTING.md)

> [!IMPORTANT]
> BrainAxL is research software and is not approved for clinical diagnosis or
> treatment. MRI data, checkpoints, and challenge artifacts are not distributed
> in this repository.

> [!NOTE]
> Selected weights are consolidated at
> [Hugging Face](https://huggingface.co/hieuphamha/BrainAxL).
> The model repository is public as of 3 September 2026.

## Method at a glance

BrainAxL uses a five-stage 3D U-Net encoder with channel widths
`[40, 80, 160, 320, 640]`. Bidirectional LSTMs scan the three spatial axes at
the two deepest stages. Their outputs are averaged, normalized, gated, and
added residually to the convolutional features.

Pretraining combines:

- full-volume MSE reconstruction from a 60% block-masked and intensity-corrupted input; and
- VICReg-inspired variance/covariance regularization of a 512-dimensional projection of multiscale pooled features.

The reported objective is

```text
L = L_reconstruction + 0.01 * (L_variance + 0.04 * L_covariance)
```

There are no negative pairs, contrastive queue, or teacher network in the
reported checkpoint.

## Repository layout

```text
asparagus/asparagus/modules/networks/brainaxl.py
    Canonical BrainAxL architecture.

asparagus/asparagus/modules/lightning_modules/self_supervised.py
    Reconstruction and variance/covariance pretraining objective.

asparagus/configs/projects/brainaxl/
    Complete pretraining and fine-tuning recipes.

asparagus_preprocessing/
    Dataset conversion and preprocessing tools.

fomo26/submissions/
    Versioned adaptation and exact inference code for validated TEST artifacts.

scripts/verify_release.py
    Dependency-free provenance, syntax, and repository-hygiene gate.
```

The older internal name `dolphins_xlstm_unet` is retained only for checkpoint
compatibility. New work should import `asparagus.modules.networks.brainaxl`.

## Installation

Python 3.11 and a CUDA-capable PyTorch installation are recommended. The
reported run used PyTorch 2.6, CUDA 12.4, Lightning 2.4, and bfloat16 mixed
precision.

```bash
git clone https://github.com/hieuphamha19/BrainAxL.git brainaxl
cd brainaxl

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'asparagus[test]'
python -m pip install -e asparagus_preprocessing
```

Create `asparagus/.env` from the example and replace every placeholder with an
absolute path:

```bash
cp asparagus/.env.example asparagus/.env
```

At minimum, `ASPARAGUS_DATA`, `ASPARAGUS_MODELS`, `ASPARAGUS_RESULTS`,
`ASPARAGUS_CONFIGS`, and `ASPARAGUS_RAW_LABELS` must be set. Raw MRI data,
processed data, labels, and checkpoints are intentionally not committed.

## Data contract

Training reads an Asparagus task directory from
`$ASPARAGUS_DATA/<TASK_NAME>`. It must contain `dataset.json`, `paths.json`, and
the split JSON referenced by the selected config. Use the tools in
`asparagus_preprocessing/` to build these files from data you are authorized to
use.

The released pretraining config expects:

```text
$ASPARAGUS_DATA/PT903_FOMO300K_FULL_COV64_R2/
  dataset.json
  paths.json
  split_95_5_0.json
```

FOMO300K and downstream challenge data remain subject to their original access
terms. This repository does not redistribute them.

## Self-supervised pretraining

The checked-in recipe is the single-model recipe recovered from the reported
checkpoint: one GPU, batch size 128, 64³ crops, 34,660 optimizer updates, AdamW,
and bfloat16 mixed precision.

```bash
cd asparagus
asp_pretrain --config-name projects/brainaxl/pretrain
```

For a smaller GPU, preserve the effective optimizer batch with accumulation:

```bash
asp_pretrain --config-name projects/brainaxl/pretrain \
  training.batch_size=16 training.accumulate_grad_batches=8
```

This is a memory-equivalent convenience setting, not the exact reported run.
The default recipe requires approximately 56 GB of GPU memory.

## Fine-tuning

The exact adaptation and inference code recovered from the six validated
FOMO26 TEST submissions is available in
[`fomo26/submissions/`](fomo26/submissions/README.md). That release maps every
submission ID to its submitted SIF, training recipe, inference entrypoint, and
artifact checksum. It is the authoritative reference for submissions
`9777066`–`9777071`.

For an end-to-end explanation of checkpoint initialization, full fine-tuning,
frozen probing, checkpoint selection, and the exact submitted recipes, see the
dedicated [training guide](docs/training.md). Sanitized per-epoch logs and
machine-readable run summaries are published in
[`training_logs/`](training_logs/README.md).

The commands below are reusable public baseline templates. They are not a
replacement for the versioned submission-specific recipes above.

All downstream runs should start from the same pretrained checkpoint. Pass an
absolute path so the run is independent of local run-ID databases.

Classification:

```bash
cd asparagus
asp_finetune_cls --config-name projects/brainaxl/finetune_cls \
  checkpoint_path=/absolute/path/to/brainaxl.ckpt
```

Regression:

```bash
asp_finetune_reg --config-name projects/brainaxl/finetune_reg \
  checkpoint_path=/absolute/path/to/brainaxl.ckpt
```

Segmentation:

```bash
asp_finetune_seg --config-name projects/brainaxl/finetune_seg \
  checkpoint_path=/absolute/path/to/brainaxl.ckpt
```

The defaults target FOMO26 Tasks 1, 3, and 2, respectively. Reuse the same
entrypoints for Tasks 4 and 5 by overriding the task and split names:

```bash
asp_finetune_seg --config-name projects/brainaxl/finetune_seg \
  checkpoint_path=/absolute/path/to/brainaxl.ckpt \
  task=SEG010_FOMO26_TrigeminalNeuralgia stem=brainaxl_task4

asp_finetune_cls --config-name projects/brainaxl/finetune_cls \
  checkpoint_path=/absolute/path/to/brainaxl.ckpt \
  task=CLS003_FOMO26_Polymicrogyria stem=brainaxl_task5
```

The classification/regression recipe encodes each MRI sequence independently,
pools mean and maximum features from all five encoder scales, fuses sequences
with a learned gate, projects the resulting 2,480-dimensional vector to 512
dimensions, and attaches a task head. Segmentation uses the pretrained encoder
and a U-Net decoder.

## Use BrainAxL as an encoder

```python
import torch

from asparagus.modules.networks.brainaxl import brainaxl_b

model = brainaxl_b(input_channels=1, output_channels=1).eval()
image = torch.randn(1, 1, 64, 64, 64)

with torch.inference_mode():
    reconstruction, embedding, feature_maps = model.forward_with_multiscale_features(image)

assert embedding.shape == (1, 2480)
```

Load released Lightning checkpoints through the fine-tuning commands above;
they handle the `model.` prefix and exclude the pretraining decoder when
requested.

## Verification

Run the dependency-free release-integrity gate from the repository root:

```bash
python scripts/verify_release.py
```

Then run the lightweight architecture and compatibility tests:

```bash
PYTHONPATH=asparagus python -m pytest asparagus/tests/test_brainaxl.py
```

Equivalently, `make check` runs both gates. The test suite checks tensor
shape/gradient behavior, published parameter counts, and state-dictionary
compatibility with checkpoints saved under the historical model name. Root CI
runs these checks for every push and pull request.

## Reproducibility details

| Setting | Reported value |
|---|---:|
| Input | `1 × 64 × 64 × 64` |
| Encoder parameters | 31,837,640 |
| Reconstruction decoder parameters | 13,196,241 |
| Semantic projector parameters | 1,274,720 |
| Total active pretraining parameters | 46,308,601 |
| Mask | 60% of non-overlapping `4 × 4 × 4` blocks |
| Optimizer | AdamW, lr `1e-4`, weight decay `3e-5`, betas `(0.9, 0.98)` |
| Schedule | cosine decay, no warm-up |
| Updates | 34,660 |
| Precision | bfloat16 mixed |
| Hardware | one NVIDIA H100 80 GB per model |

Additional methodological details are documented in
[`BrainAxL_SSL_Framework.md`](BrainAxL_SSL_Framework.md).

For artifact boundaries, checkpoint contracts, and the distinction between
generic configs and exact TEST submissions, see the
[reproducibility guide](docs/reproducibility.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff), which GitHub
exposes through **Cite this repository**. Paper-specific authors, venue, and DOI
will be added when finalized.

## Project policies

- Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security and clinical-use notice: [`SECURITY.md`](SECURITY.md)
- Community expectations: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)

## License and attribution

BrainAxL is distributed under the Apache License 2.0; see [`LICENSE`](LICENSE).
The vendored Asparagus code retains its license in
[`asparagus/LICENSE`](asparagus/LICENSE). Please also cite the upstream
Asparagus project when using this implementation. Attribution details are in
[`NOTICE`](NOTICE).
