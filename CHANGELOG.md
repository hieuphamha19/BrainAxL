# Changelog

All notable public-release changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

## [Unreleased]

### Added

- End-to-end pretraining, full fine-tuning, frozen-probe, and frozen-embedding
  documentation.
- Sanitized pretraining and downstream epoch logs, selected fold metrics, and
  a machine-readable run index for submissions 9777066–9777071.
- A single organized Hugging Face weight repository, download guide, and
  checksum manifest, published manually after the challenge embargo.

## [1.0.0] - 2026-08-28

### Added

- Canonical BrainAxL architecture and checkpoint-compatible legacy aliases.
- Recovered self-supervised pretraining and downstream fine-tuning configs.
- Exact inference payloads and clean adaptation recipes for FOMO26 TEST
  submissions 9777066–9777071.
- Machine-readable SIF/source provenance and integrity checks.
- Root-level CI, contribution guidelines, security policy, citation metadata,
  and GitHub issue/PR templates.

### Changed

- Public documentation now uses the BrainAxL name consistently while retaining
  historical module names only for checkpoint compatibility.

[Unreleased]: https://github.com/hieuphamha19/BrainAxL/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hieuphamha19/BrainAxL/releases/tag/v1.0.0
