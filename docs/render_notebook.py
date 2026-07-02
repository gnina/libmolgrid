#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


def sentence_case_title(stem: str) -> str:
    return stem.replace("_", " ").lower().capitalize()


def render_notebook(source: Path, destination: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(notebook)

    body = body.replace(
        '<div class="container" id="notebook-container">',
        '<div class="bs-docs-container row">\n<div class="col-md-9" role="main">',
    )
    body = body.replace("</body>", "\n</div>\n</body>")
    body = re.sub(r"(<img[^>]*src[^>]*>)<br>", r"\1", body)
    body = body.replace("<pre>", '<pre style=" white-space: pre;">')

    front_matter = (
        "---\n"
        f"title: {sentence_case_title(source.stem)}\n"
        "layout: tutorials\n"
        "homepage: false\n"
        "hide: true\n"
        "---\n\n"
    )

    destination.write_text(front_matter + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a notebook to Jekyll-compatible HTML.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    render_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
