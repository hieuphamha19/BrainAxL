# Contributing to BrainAxL

Thank you for helping improve BrainAxL. Contributions that strengthen
reproducibility, documentation, tests, portability, or the model implementation
are welcome.

## Before opening a change

1. Search existing issues and pull requests.
2. Use an issue for behavior changes or substantial design proposals.
3. Never commit patient data, challenge data, model checkpoints, credentials,
   local absolute paths, or derived artifacts that may contain sensitive data.
4. Keep changes focused and explain their scientific or engineering rationale.

## Development setup

```bash
git clone https://github.com/hieuphamha19/BrainAxL.git
cd BrainAxL
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'asparagus[test]'
```

Run the release-integrity and model tests before submitting:

```bash
python scripts/verify_release.py
PYTHONPATH=asparagus python -m pytest asparagus/tests/test_brainaxl.py
```

## Pull requests

- Add or update tests for behavior changes.
- Update README/config documentation when interfaces or recipes change.
- Preserve backwards compatibility for published checkpoint keys unless the
  pull request explicitly documents a migration.
- Use clear commit messages and complete the pull-request checklist.

## Immutable submission payloads

Files listed in `fomo26/submissions/sources.sha256` reproduce source embedded in
validated TEST SIFs. Do not format or refactor them in place. Put improvements
in a new versioned directory and retain the original payload plus checksum for
provenance.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0.
