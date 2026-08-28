## Summary

Describe what changed and why.

## Validation

List the commands or experiments used to validate the change.

## Checklist

- [ ] The change is focused and documented.
- [ ] Tests were added or updated when behavior changed.
- [ ] `python scripts/verify_release.py` passes.
- [ ] No patient data, restricted challenge data, checkpoints, credentials, or
      machine-specific absolute paths are included.
- [ ] Published checkpoint compatibility is preserved or migration steps are
      documented.
- [ ] Files protected by `fomo26/submissions/sources.sha256` are unchanged, or a
      new versioned artifact with provenance was added.
