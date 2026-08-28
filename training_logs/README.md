# Training logs

This directory contains sanitized records for the BrainAxL checkpoint and the
adaptation runs used by the validated FOMO26 TEST submissions. The original
Lightning logs were reduced to the metrics needed to audit optimization and
checkpoint selection. Absolute storage paths, user names, data identifiers,
checkpoints, predictions, and restricted challenge files are excluded.

| Artifact | Adaptation | Public record |
|---:|---|---|
| BrainAxL run 19726 | self-supervised pretraining | [20 epoch snapshots](pretraining_run19726_epochs.csv) |
| 9777066 / Task 1 | frozen encoder, two logistic heads | [run summary](runs.json) |
| 9777071 / Task 2 | five-fold full fine-tuning | [selected fold checkpoints](task2_9777071_folds.csv) |
| 9777067 / Task 3 | five-fold full fine-tuning | [300 epoch records](task3_9777067_epochs.csv) |
| 9777068 / Task 4 | one-fold full fine-tuning | [200 epoch records](task4_9777068_epochs.csv) |
| 9777069 / Task 5 | frozen encoder, logistic head | [run summary](runs.json) |
| 9777070 / Tasks 6/7 | frozen embedding extractor | [run summary](runs.json) |

[runs.json](runs.json) is the machine-readable index for configurations,
selection rules, best validation metrics, and log provenance. The CSV files
preserve the wall-clock strings emitted by the original logger; the source did
not record a timezone. Empty Task 4 validation cells are expected because that
run validated every two epochs.

Pretraining values are the last metric snapshot emitted within each epoch, not
retrospectively recomputed dataset averages. Task 3 and Task 4 values are the
epoch-level values emitted by Lightning. Task 2's original epoch logs were not
retained as one homogeneous public series because the submitted ensemble
combines two confirmation folds and three original-run folds; the selected
checkpoint manifest is published instead.

Tasks 1 and 5 use deterministic scikit-learn fitting rather than iterative
neural-network optimization, so an epoch CSV would be artificial. Submission
9777070 performs no label training at all: the challenge evaluator fits its
own probe from the submitted frozen embeddings.

All metrics in this directory are training or local-validation records. They
are not hidden TEST labels or official leaderboard scores.

