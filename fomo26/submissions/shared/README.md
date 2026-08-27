# Shared segmentation fine-tuning wrapper

`run_finetune_seg_small_object.py` is the training wrapper used for the Task 2
and Task 4 small-object recipes. It installs foreground-aware MONAI sampling,
the generalized-Dice/focal loss, full-volume validation, post-processing, and
the validation metric used for checkpoint selection before calling the
Asparagus segmentation entrypoint.

The two adjacent modules implement optional Task 2 acquisition augmentations.
They remain included because the wrapper imports them, although all such
options are disabled in the submitted recipe.
