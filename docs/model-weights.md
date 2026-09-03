# Model weights

The canonical BrainAxL foundation checkpoint is published at:

**[hieuphamha/BrainAxL](https://huggingface.co/hieuphamha/BrainAxL)**

The model repository was made public on 3 September 2026, after the official
FOMO26 TEST deadline of 30 August 2026 at 11:59 PM Pacific. It contains only
the self-supervised BrainAxL-B checkpoint from run 19726. Downstream,
submission-specific, and SIF artifacts are not included in this public model
release.

## Artifact layout

```text
brainaxl-b/
    model.safetensors
    model.ckpt
    training_config.yaml
    hparams.yaml
    hydra_overrides.yaml
config.json
preprocessing.json
metadata.json
checksums.sha256
examples/load_weights.py
```

`model.safetensors` is the canonical safe tensor-only state dictionary.
`model.ckpt` contains the same tensors in a weight-only PyTorch Lightning
wrapper for compatibility with the reported Asparagus pipeline. Both preserve
the historical state-dictionary names used by run 19726.

## Download

The safe format is recommended for new integrations:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

path = hf_hub_download(
    repo_id="hieuphamha/BrainAxL",
    filename="brainaxl-b/model.safetensors",
)
state_dict = load_file(path, device="cpu")
```

Existing Asparagus pipelines can download the compatibility checkpoint:

```python
checkpoint = hf_hub_download(
    repo_id="hieuphamha/BrainAxL",
    filename="brainaxl-b/model.ckpt",
)
```

Verify downloaded files against `checksums.sha256`. Architecture,
preprocessing, provenance, sizes, and hashes are also recorded in the JSON
metadata files.

## Release scope

The GitHub repository continues to publish downstream training code,
configurations, sanitized logs, evaluation methodology, and submission
provenance. Task-specific trained weights are kept out of the public
foundation-model repository so the release has one clear purpose and model
card.

No Hugging Face token is stored in GitHub. The archived model repository that
previously contained mixed artifacts and the separate SIF repository remain
private.
