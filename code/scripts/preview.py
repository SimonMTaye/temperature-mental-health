"""Command-line entrypoint for rendering the results preview."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PREVIEW_DIR = PROJECT / "code" / "preview"
DOCS_OUTPUT = PROJECT / "docs" / "index.html"
RENDERED_PREVIEW = PREVIEW_DIR / "preview.html"


def render_preview(*, stage: bool = False) -> None:
    """Render preview.qmd to docs/index.html, optionally staging the result."""
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT.unlink(missing_ok=True)
    RENDERED_PREVIEW.unlink(missing_ok=True)

    subprocess.run(
        ["quarto", "render", "preview.qmd", "--to", "html", "--output", "preview.html"],
        cwd=PREVIEW_DIR,
        check=True,
    )
    shutil.move(RENDERED_PREVIEW, DOCS_OUTPUT)

    if not DOCS_OUTPUT.exists():
        raise FileNotFoundError(f"expected rendered preview at {DOCS_OUTPUT}")

    if stage:
        subprocess.run(["git", "add", str(DOCS_OUTPUT.relative_to(PROJECT))], cwd=PROJECT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Quarto preview page.")
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage docs/index.html after rendering, matching the pre-commit hook.",
    )
    args = parser.parse_args()
    render_preview(stage=args.stage)


if __name__ == "__main__":
    main()
