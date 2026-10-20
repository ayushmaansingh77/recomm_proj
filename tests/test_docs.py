"""Test suite for Documentation Integrity and CLI Contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def test_documented_files_exist():
    """Verify that all files documented in README.md and interview.md exist in codebase."""
    documented_files = [
        "config.py",
        "train.py",
        "app.py",
        "theme.py",
        "components.py",
        "requirements.txt",
        "README.md",
        "interview.md",
        ".gitignore",
        "src/__init__.py",
        "src/data.py",
        "src/models.py",
        "src/evaluate.py",
        "src/pipeline.py",
        "src/recommender.py",
        "views/__init__.py",
        "views/for_you.py",
        "views/item_cold_start.py",
        "views/under_the_hood.py",
        "views/item_detail.py",
    ]

    for rel_path in documented_files:
        full_path = BASE_DIR / rel_path
        assert full_path.exists(), f"Documented file '{rel_path}' does not exist on disk"


def test_cli_help_flag():
    """Verify that train.py CLI interface responds to --help as documented."""
    res = subprocess.run(
        [sys.executable, str(BASE_DIR / "train.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )
    assert res.returncode == 0, f"train.py --help failed: {res.stderr}"
    assert "--category" in res.stdout
    assert "--max-reviews" in res.stdout
    assert "--eval-users" in res.stdout
    assert "--embedder" in res.stdout
    assert "--synthetic" in res.stdout


def test_interview_md_sections():
    """Verify that interview.md contains all primary architectural and file walkthrough sections."""
    interview_path = BASE_DIR / "interview.md"
    assert interview_path.exists(), "interview.md is missing"
    content = interview_path.read_text(encoding="utf-8")

    expected_sections = [
        "Executive Summary",
        "Dual Feature Space",
        "Adaptive Blending Formula",
        "System Architecture Flow",
        "config.py",
        "train.py",
        "app.py",
        "theme.py",
        "components.py",
        "views/",
        "src/data.py",
        "src/models.py",
        "src/evaluate.py",
        "src/pipeline.py",
        "src/recommender.py",
    ]

    for section in expected_sections:
        assert section in content, f"Expected section '{section}' missing from interview.md"


def test_readme_md_sections():
    """Verify that README.md contains expected setup and deployment instructions."""
    readme_path = BASE_DIR / "README.md"
    assert readme_path.exists(), "README.md is missing"
    content = readme_path.read_text(encoding="utf-8")
    assert "Cold-Start Recommender" in content
    assert "Streamlit" in content
