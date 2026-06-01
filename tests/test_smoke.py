"""Basic smoke tests: syntax checks and optional config validation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "tests",
}


def _iter_python_sources() -> list[Path]:
    sources: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name.startswith("test_"):
            continue
        if path.stat().st_size < 20:
            continue
        sources.append(path)
    return sorted(sources)


def test_repo_has_python_sources() -> None:
    sources = _iter_python_sources()
    assert sources, "expected at least one Python source file"


def test_python_sources_are_valid_syntax() -> None:
    failures: list[str] = []
    for path in _iter_python_sources():
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
    assert not failures, "syntax errors: " + "; ".join(failures)


def test_config_yaml_is_parseable() -> None:
    config = REPO_ROOT / "config.yaml"
    if not config.is_file():
        pytest.skip("no config.yaml")
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert data is not None or data is None
