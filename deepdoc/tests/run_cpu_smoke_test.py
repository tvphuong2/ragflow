#!/usr/bin/env python3
"""Utility to exercise the DeepDoc pipeline on CPU-only environments.

This script disables GPU execution providers before instantiating the
``RAGFlowPdfParser`` so that the ONNX Runtime and auxiliary components run on
CPU.  It prints a short summary for each parsed document, which makes it useful
for smoke tests in CI or while validating a CPU-only deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import Iterable

try:
    from deepdoc.parser import PdfParser as _PdfParser
except Exception as exc:  # pragma: no cover - import errors should surface early
    raise RuntimeError("Failed to import DeepDoc PDF parser") from exc


def _ensure_cpu_environment() -> None:
    """Force DeepDoc and dependencies to stay on CPU.

    We clear ``CUDA_VISIBLE_DEVICES`` and, when possible, set thread-related
    environment variables so the run is deterministic for small smoke tests.
    The ``LAYOUT_RECOGNIZER_TYPE`` flag defaults to the ONNX pipeline, which is
    the only implementation that works without a dedicated accelerator.
    """

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


def _summarise_boxes(boxes: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    preview = []
    for box in boxes[:limit]:
        preview.append(
            {
                "page": int(box.get("page_number", -1)),
                "layout_type": box.get("layout_type"),
                "text": str(box.get("text", "")).strip().splitlines()[0:3],
            }
        )
    return preview


def run_smoke_test(pdf_paths: Iterable[str], *, limit: int = 3, zoom: int = 3) -> None:
    _ensure_cpu_environment()
    logging.info("Initialising DeepDoc PDF parser (CPU mode)…")
    parser = _PdfParser()

    for pdf_path in _iter_pdf_paths(pdf_paths):
        logging.info("Processing %s", pdf_path)
        started = time.time()
        boxes = parser.parse_into_bboxes(str(pdf_path), zoomin=zoom)
        elapsed = time.time() - started
        logging.info(
            "Parsed %s in %.2fs -> %d blocks", pdf_path.name, elapsed, len(boxes)
        )
        summary = {
            "file": pdf_path.name,
            "blocks": len(boxes),
            "elapsed_seconds": round(elapsed, 3),
            "preview": _summarise_boxes(boxes, limit),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="+",
        help="Path(s) to PDF files that should be parsed using the CPU-only pipeline.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=3,
        help="Number of text blocks to preview for each PDF (default: 3).",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=3,
        help=(
            "Zoom factor passed to the parser. Larger values increase precision "
            "at the cost of runtime (default: 3)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    try:
        run_smoke_test(args.pdf, limit=args.preview_limit, zoom=args.zoom)
    except Exception as exc:
        logging.exception("DeepDoc CPU smoke test failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
