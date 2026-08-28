# FOMO26 TEST submission code

This directory records the code used by CVMAIL_VinUni for the final FOMO26
Method Track TEST submissions. Each `predict.py` was extracted directly from
the corresponding validated SIF; it is not a later development snapshot.

| Submission | Task | Submitted image | Adaptation |
|---:|---|---|---|
| 9777071 | Task 2: meningioma segmentation | `task2 (3).sif` (local `task2.sif`) | full fine-tuning, five folds |
| 9777070 | Tasks 6/7: probing and fairness | `task6.sif` | frozen embedding extractor; probe fitted by evaluator |
| 9777069 | Task 5: polymicrogyria classification | `task5_deconf_ap140_v3.sif` | frozen encoder + domain-augmented logistic head |
| 9777068 | Task 4: trigeminal neuralgia segmentation | `task4.sif` | full fine-tuning, one fold |
| 9777067 | Task 3: brain-age regression | `task3_v2_spacing1mm.sif` | full fine-tuning, five folds |
| 9777066 | Task 1: infarct detection | `task1_ens.sif` | frozen encoder + two-view logistic ensemble |

The common initialization is BrainAxL/xLSTM SSL checkpoint run `19726`, with
`starting_filters=40`, xLSTM stages `(3, 4)`, and a 512-dimensional semantic
projector. Checkpoints, SIF images, and challenge data are intentionally not
stored in Git. Set the checkpoint environment variable documented by each task.

Install the framework and lightweight downstream dependencies from the repo
root:

```bash
python -m pip install -e asparagus
python -m pip install -r fomo26/submissions/requirements.txt
```

## Reproduction scope

- `predict.py` files are byte-for-byte copies of the source embedded in the
  submitted images. Their hashes are listed in `sources.sha256`.
- `train_folds.sh`, `extract_features.py`, and `fit_head.py` expose the exact
  selected recipes using portable command-line arguments and relative paths.
- Tasks 1 and 5 are frozen linear probes, not end-to-end fine-tunes.
- Tasks 6 and 7 submit a 1,024-dimensional frozen representation. The official
  evaluation harness trains the downstream probe, so there is no task-label
  fine-tuning script for submission 9777070.
- The Task 2 leaderboard filename contains a browser-added suffix, while the
  audited local artifact is `task2.sif`.

Run a syntax check without checkpoints or challenge data:

```bash
python scripts/verify_release.py
```

See each task directory for inputs, checkpoint layout, training, and inference
commands. `artifact_manifest.json` provides a machine-readable mapping from
submission IDs to SIF and source hashes. The repository-wide
[`reproducibility guide`](../../docs/reproducibility.md) documents artifact
boundaries and the policy for publishing future versions.

The repository-wide [training guide](../../docs/training.md) explains the
complete optimization and adaptation flow. Sanitized epoch histories and
checkpoint-selection records are indexed in
[`training_logs/`](../../training_logs/README.md).
