#!/usr/bin/env python3
"""Verify BrainAxL release provenance without importing ML dependencies."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO_ROOT / "fomo26" / "submissions"
MANIFEST = SUBMISSIONS / "artifact_manifest.json"
SOURCE_HASHES = SUBMISSIONS / "sources.sha256"
EXPECTED_SUBMISSIONS = {
    9777066: ("task1_ens.sif", "task1_9777066/predict.py"),
    9777067: ("task3_v2_spacing1mm.sif", "task3_9777067/predict.py"),
    9777068: ("task4.sif", "task4_9777068/predict.py"),
    9777069: ("task5_deconf_ap140_v3.sif", "task5_9777069/predict.py"),
    9777070: ("task6.sif", "task67_9777070/predict.py"),
    9777071: ("task2.sif", "task2_9777071/predict.py"),
}
FORBIDDEN_RELEASE_SUFFIXES = (
    ".ckpt",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".pt",
    ".sif",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class VerificationError(RuntimeError):
    """Raised when the public release violates an integrity contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_manifest() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload.get("submissions", [])
    by_id = {int(entry["submission_id"]): entry for entry in entries}
    require(len(by_id) == len(entries), "artifact manifest has duplicate IDs")
    require(
        set(by_id) == set(EXPECTED_SUBMISSIONS),
        "artifact manifest does not contain exactly submissions 9777066-9777071",
    )

    for submission_id, (sif_name, source_name) in EXPECTED_SUBMISSIONS.items():
        entry = by_id[submission_id]
        require(entry["local_filename"] == sif_name, f"{submission_id}: wrong SIF")
        require(entry["source"] == source_name, f"{submission_id}: wrong source")
        require(
            SHA256_PATTERN.fullmatch(entry["sif_sha256"]) is not None,
            f"{submission_id}: invalid SIF SHA-256",
        )
        require(
            SHA256_PATTERN.fullmatch(entry["source_sha256"]) is not None,
            f"{submission_id}: invalid source SHA-256",
        )
        source = SUBMISSIONS / source_name
        require(source.is_file(), f"{submission_id}: missing {source_name}")
        require(
            sha256(source) == entry["source_sha256"],
            f"{submission_id}: submitted source checksum changed",
        )
    return len(entries)


def verify_source_hashes() -> int:
    seen: set[Path] = set()
    for line_number, raw_line in enumerate(
        SOURCE_HASHES.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        parts = raw_line.split(maxsplit=1)
        require(len(parts) == 2, f"sources.sha256:{line_number}: malformed line")
        expected, relative_name = parts
        relative = Path(relative_name.strip())
        require(not relative.is_absolute(), f"absolute source path: {relative}")
        require(".." not in relative.parts, f"unsafe source path: {relative}")
        require(relative not in seen, f"duplicate source path: {relative}")
        require(
            SHA256_PATTERN.fullmatch(expected) is not None,
            f"invalid SHA-256 for {relative}",
        )
        source = SUBMISSIONS / relative
        require(source.is_file(), f"missing protected source: {relative}")
        require(sha256(source) == expected, f"checksum mismatch: {relative}")
        seen.add(relative)
    require(seen, "sources.sha256 is empty")
    return len(seen)


def verify_no_large_artifacts() -> None:
    for path in SUBMISSIONS.rglob("*"):
        if path.is_file() and path.name.lower().endswith(FORBIDDEN_RELEASE_SUFFIXES):
            raise VerificationError(f"release artifact must not be committed: {path}")


def verify_python_syntax() -> int:
    count = 0
    for path in REPO_ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        compile(source, str(path.relative_to(REPO_ROOT)), "exec")
        count += 1
    return count


def verify_shell_syntax() -> int:
    scripts = sorted(SUBMISSIONS.rglob("*.sh"))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    return len(scripts)


def verify_json() -> int:
    count = 0
    for path in REPO_ROOT.rglob("*.json"):
        if ".git" in path.parts:
            continue
        json.loads(path.read_text(encoding="utf-8"))
        count += 1
    return count


def verify_local_markdown_links() -> int:
    count = 0
    owned_markdown = [
        *REPO_ROOT.glob("*.md"),
        *(REPO_ROOT / "docs").rglob("*.md"),
        *(REPO_ROOT / "fomo26").rglob("*.md"),
    ]
    for markdown in owned_markdown:
        for target in MARKDOWN_LINK.findall(markdown.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if not relative_target:
                continue
            resolved = (markdown.parent / relative_target).resolve()
            require(
                resolved.exists(),
                f"broken link in {markdown.relative_to(REPO_ROOT)}: {target}",
            )
            count += 1
    return count


def verify_repository_identity() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    require("trongduc-nguyen/fomo" not in readme, "README contains stale repo URL")
    require("hieuphamha19/BrainAxL" in readme, "README missing canonical repo URL")


def main() -> int:
    try:
        submissions = verify_artifact_manifest()
        protected_sources = verify_source_hashes()
        verify_no_large_artifacts()
        python_files = verify_python_syntax()
        shell_files = verify_shell_syntax()
        json_files = verify_json()
        local_links = verify_local_markdown_links()
        verify_repository_identity()
    except (
        OSError,
        ValueError,
        VerificationError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1

    print(
        "release verification passed: "
        f"{submissions} submissions, {protected_sources} protected sources, "
        f"{python_files} Python files, {shell_files} shell scripts, "
        f"{json_files} JSON files, {local_links} local Markdown links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
