# Model weights

All selected BrainAxL weights are organized in one Hugging Face model
repository:

**[hieuphamha/BrainAxL](https://huggingface.co/hieuphamha/BrainAxL)**

The repository is private during the FOMO26 TEST embargo. The official
deadline is 30 August 2026 at 11:59 PM Pacific. Publication must occur only
after 31 August 2026 at 08:00 UTC and after a final disclosure audit.

Only this foundation-model repository is scheduled for publication. The
separate private repository containing submitted SIF images is outside the
workflow's scope and remains private.

## Artifact layout

```text
pretraining/brainaxl-b-run19726/
    last.ckpt
finetuning/task1-9777066/
    task1_head_params_3ch_ens_mm_s3.npz
finetuning/task2-9777071/
    fold0_best.ckpt ... fold4_best.ckpt
    inference_config.json
finetuning/task3-9777067/
    fold0_best.ckpt ... fold4_best.ckpt
finetuning/task4-9777068/
    best.ckpt
finetuning/task5-9777069/
    task5_head_params_t1_ap140_s3mean_domainaug_C0.05.npz
```

Tasks 1 and 5 reuse the frozen run-19726 encoder and load their small linear
heads. Tasks 2–4 load task-specific neural-network checkpoints. Tasks 6/7 use
the frozen run-19726 checkpoint directly and therefore have no additional
task-trained weight file.

Downstream checkpoints are weight-only exports: they retain the exact
`state_dict` used for inference while excluding optimizer and scheduler
states. The HF repository includes `weights_manifest.json` with byte sizes,
SHA-256 hashes, submission IDs, task IDs, and folds, plus
`weights_checksums.sha256`.

## Download

After the embargo, download only the artifact required for a task:

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="hieuphamha/BrainAxL",
    filename="pretraining/brainaxl-b-run19726/last.ckpt",
)
```

Before public release, authenticated team members can use the same call after
logging in with a Hugging Face token that has access to the private repo.

The historical `lstm-s512/` and `resencb-s512/` paths remain in the model repo
for compatibility. New BrainAxL code should use the organized `pretraining/`
and `finetuning/` paths.

## Embargo automation

The [publication workflow](../.github/workflows/publish-hf-weights.yml) targets
only `hieuphamha/BrainAxL` and refuses to change visibility
before `2026-08-31T08:00:00Z`. It never targets the SIF repository.
