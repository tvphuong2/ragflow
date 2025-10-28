#!/usr/bin/env python3
"""Exercise the DeepDoc HTML-aware parser in CPU-friendly environments.

The tool parses each provided PDF with :class:`~rag.app.laws_html.PdfDeepDocHTML`
so that heading detection and layout-derived metadata can be inspected without a
GPU.  It mirrors the CPU guard rails used by ``run_cpu_smoke_test`` and prints a
JSON summary for quick verification in CI or during manual debugging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import Iterable, Optional

from rag.app.laws_html import HtmlAwareSection, PdfDeepDocHTML


def _ensure_cpu_environment() -> None:
    """Force DeepDoc dependencies to stick to CPU execution providers."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("LAYOUT_RECOGNIZER_TYPE", "onnx")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _iter_pdf_paths(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    for raw in paths:
        path = pathlib.Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            logging.warning("%s does not have a .pdf extension", path)
        yield path


def _summarise_sections(
    sections: list[HtmlAwareSection], limit: int
) -> list[dict[str, Optional[str]]]:
    preview: list[dict[str, Optional[str]]] = []
    if limit <= 0:
        return preview

    for section in sections[:limit]:
        text_lines = section.text.strip().splitlines()
        preview.append(
            {
                "level": section.level if section.level is not None else None,
                "layout_type": section.layout_type or None,
                "snippet": " ".join(text_lines[:2])[:240] if text_lines else "",
                "tags": section.tags or None,
            }
        )
    return preview


def _count_heading_levels(sections: list[HtmlAwareSection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for section in sections:
        if section.level is None:
            continue
        key = str(section.level)
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_html_smoke_test(
    pdf_paths: Iterable[str], *, limit: int = 5, zoom: int = 3
) -> None:
    _ensure_cpu_environment()
    logging.info("Initialising DeepDoc HTML parser (CPU mode)…")
    parser = PdfDeepDocHTML()

    def _progress_callback(*args, **kwargs) -> None:
        progress = None
        message = kwargs.get("msg")
        if args:
            progress = args[0]
            if len(args) > 1 and message is None:
                message = args[1]
        if message:
            logging.info("[DeepDoc] %s", message)
        if isinstance(progress, (int, float)):
            logging.debug("[DeepDoc] progress %.2f", float(progress))

    for pdf_path in _iter_pdf_paths(pdf_paths):
        logging.info("Processing %s", pdf_path)
        started = time.time()
        sections, metadata = parser(
            str(pdf_path),
            from_page=0,
            to_page=100000,
            zoomin=zoom,
            callback=_progress_callback,
        )
        elapsed = time.time() - started
        logging.info(
            "Parsed %s in %.2fs -> %d sections", pdf_path.name, elapsed, len(sections)
        )
        heading_distribution = _count_heading_levels(sections)
        if heading_distribution:
            logging.info(
                "[LawsHTML] Heading distribution for %s: %s",
                pdf_path.name,
                heading_distribution,
            )
        else:
            logging.info(
                "[LawsHTML] No heading levels detected for %s", pdf_path.name
            )
        summary = {
            "file": pdf_path.name,
            "sections": len(sections),
            "heading_levels": heading_distribution,
            "elapsed_seconds": round(elapsed, 3),
            "preview": _summarise_sections(sections, limit),
        }
        if metadata:
            summary["metadata"] = metadata
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="+",
        help=(
            "Path(s) to PDF files that should be parsed with the DeepDoc HTML "
            "layout pipeline."
        ),
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=5,
        help="Number of sections to preview for each PDF (default: 5).",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=3,
        help=(
            "Zoom factor passed to DeepDoc. Higher values increase precision at "
            "the cost of runtime (default: 3)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    try:
        run_html_smoke_test(
            args.pdf,
            limit=args.preview_limit,
            zoom=args.zoom,
        )
    except Exception as exc:  # pragma: no cover - surface CLI failures
        logging.exception("DeepDoc HTML smoke test failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
