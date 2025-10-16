#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Optional

from api.db import ParserType
from rag.app import laws
from rag.app.policy import (
    FrontLine,
    LineEntry,
    _build_front_matter_chunks,
    _build_markdown_chunks,
    _determine_heading_threshold,
    _extract_page_from_tags,
)
from rag.nlp import (
    bullets_category,
    rag_tokenizer,
    tokenize_chunks,
)


@dataclass
class HtmlAwareSection:
    text: str
    tags: str
    layout_type: str
    level: Optional[int]


class PdfDeepDocHTML(laws.Pdf):
    """DeepDoc variant that preserves layout hints for downstream HTML heuristics."""

    _LAYOUT_LEVEL_HINTS = (
        ("title", 1),
        ("chapter", 2),
        ("header", 2),
        ("heading", 2),
        ("section", 3),
        ("subsection", 4),
        ("subheading", 4),
        ("article", 3),
        ("clause", 4),
    )

    def __init__(self) -> None:
        super().__init__()
        self.model_speciess = ParserType.LAWS_HTML.value

    def __call__(
        self,
        filename,
        binary=None,
        from_page: int = 0,
        to_page: int = 100000,
        zoomin: int = 3,
        callback=None,
    ) -> tuple[List[HtmlAwareSection], Optional[dict]]:
        raw_sections, metadata = super().__call__(
            filename,
            binary=binary,
            from_page=from_page,
            to_page=to_page,
            zoomin=zoomin,
            callback=callback,
        )
        boxes = list(getattr(self, "boxes", []))
        body_indents = [
            b.get("x0", 0.0)
            for b in boxes
            if self._is_body_candidate(b)
        ]
        body_indent = median(body_indents) if body_indents else 0.0

        sections: List[HtmlAwareSection] = []
        for idx, (text, tags) in enumerate(raw_sections):
            clean_text = text.strip()
            if not clean_text:
                continue

            layout_type = ""
            level: Optional[int] = None
            if idx < len(boxes):
                layout_type = str(boxes[idx].get("layout_type", ""))
                level = self._estimate_heading_level(
                    boxes[idx], clean_text, body_indent
                )

            sections.append(
                HtmlAwareSection(
                    text=clean_text,
                    tags=tags or "",
                    layout_type=layout_type,
                    level=level,
                )
            )

        return sections, metadata

    @staticmethod
    def _is_body_candidate(box: dict) -> bool:
        layout = str(box.get("layout_type", "")).lower()
        if not layout:
            return True
        if any(key in layout for key in ("title", "head", "foot", "caption")):
            return False
        return True

    def _estimate_heading_level(
        self,
        box: dict,
        clean_text: str,
        body_indent: float,
    ) -> Optional[int]:
        layout = str(box.get("layout_type", "")).lower()
        for key, level in self._LAYOUT_LEVEL_HINTS:
            if key in layout:
                return min(level, 6)

        inferred = self._infer_numeric_heading(clean_text)
        if inferred is not None:
            return min(inferred, 6)

        indent = float(box.get("x0", 0.0) or 0.0)
        if body_indent and indent <= body_indent * 0.85:
            return 2

        condensed = re.sub(r"\s+", "", clean_text)
        if condensed.isupper() and len(condensed) <= 80:
            return 2

        if len(clean_text.split()) <= 6 and clean_text.endswith(":"):
            return 3

        return None

    @staticmethod
    def _infer_numeric_heading(text: str) -> Optional[int]:
        normalized = text.strip()
        match = re.match(r"^([0-9]+(?:\.[0-9]+)*)", normalized)
        if not match:
            return None
        segments = match.group(1).split(".")
        return min(len(segments), 6)


def _build_lines(sections: Iterable[HtmlAwareSection]) -> List[LineEntry]:
    lines: List[LineEntry] = []
    for section in sections:
        level = section.level
        if level is not None and level <= 0:
            level = None
        lines.append(
            LineEntry(
                level=level,
                text=section.text,
                tags=section.tags,
            )
        )
    return lines


def _build_front_lines(sections: Iterable[HtmlAwareSection]) -> List[FrontLine]:
    front_lines: List[FrontLine] = []
    for order, section in enumerate(sections):
        page = _extract_page_from_tags(section.tags)
        front_lines.append(
            FrontLine(order=order, text=section.text, tags=section.tags, page=page)
        )
    return front_lines


def _compute_heading_threshold(
    lines: Iterable[LineEntry],
    bull: int,
) -> Optional[int]:
    candidate_levels = [line.level for line in lines if line.level is not None]
    if not candidate_levels:
        return _determine_heading_threshold((line.level for line in lines), bull)

    levels_sorted = sorted(set(candidate_levels))
    if len(levels_sorted) == 1:
        return levels_sorted[0]
    return min(levels_sorted[1], 6)


def chunk(
    filename,
    binary=None,
    from_page: int = 0,
    to_page: int = 100000,
    lang: str = "Chinese",
    callback=None,
    **kwargs,
):
    callback = callback or (lambda *args, **kw: None)

    parser_config = kwargs.get(
        "parser_config",
        {
            "chunk_token_num": 512,
            "delimiter": "\n!?。；！？",
            "layout_recognize": "DeepDocHTML",
        },
    )
    chunk_token_num = parser_config.get("chunk_token_num", 512)

    doc = {"docnm_kwd": filename}
    title = re.sub(r"\.[a-zA-Z0-9]+$", "", filename)
    doc["title_tks"] = rag_tokenizer.tokenize(title)
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    eng = lang.lower() == "english"

    if not re.search(r"\.pdf$", filename, re.IGNORECASE):
        logging.info(
            "LawsHTML chunker received non-PDF input. Falling back to legacy laws parser."
        )
        return laws.chunk(
            filename,
            binary=binary,
            from_page=from_page,
            to_page=to_page,
            lang=lang,
            callback=callback,
            **kwargs,
        )

    layout_choice_raw = parser_config.get("layout_recognize", "DeepDocHTML")
    layout_choice = str(layout_choice_raw or "DeepDocHTML")
    if layout_choice.lower() != "deepdochtml":
        logging.info(
            "LawsHTML chunker requires DeepDocHTML parser. Got '%s', using legacy pipeline instead.",
            layout_choice,
        )
        return laws.chunk(
            filename,
            binary=binary,
            from_page=from_page,
            to_page=to_page,
            lang=lang,
            callback=callback,
            **kwargs,
        )

    callback(0.1, "Start to parse.")
    pdf_parser = PdfDeepDocHTML()
    sections_data, _ = pdf_parser(
        filename if not binary else binary,
        from_page=from_page,
        to_page=to_page,
        callback=callback,
    )

    if not sections_data:
        callback(0.99, "No chunk parsed out.")
        return tokenize_chunks([], doc, eng, pdf_parser)

    bull = bullets_category([section.text for section in sections_data])
    lines = _build_lines(sections_data)
    front_lines = _build_front_lines(sections_data)

    heading_threshold = _compute_heading_threshold(lines, bull)

    front_chunks = _build_front_matter_chunks(front_lines, title)
    body_chunks = _build_markdown_chunks(
        lines,
        heading_threshold,
        title,
        chunk_token_num,
    )

    chunks = [ck for ck in front_chunks + body_chunks if ck and ck.strip()]
    if not chunks:
        callback(0.99, "No chunk parsed out.")

    return tokenize_chunks(chunks, doc, eng, pdf_parser)


__all__ = ["chunk", "PdfDeepDocHTML", "HtmlAwareSection"]
