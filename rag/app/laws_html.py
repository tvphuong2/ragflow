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
from collections import Counter
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
    level_hint: Optional[int] = None
    page: Optional[int] = None
    indent: float = 0.0
    font_size: float = 0.0
    height: float = 0.0
    gap_above: Optional[float] = None
    level: Optional[int] = None


@dataclass
class HeadingCandidate:
    index: int
    normalized: str
    style_key: str
    numeric_depth: Optional[int]
    features: frozenset[str]
    page: Optional[int]
    assigned_level: Optional[int] = None


@dataclass(frozen=True)
class BodyStats:
    font_median: float
    height_median: float
    gap_median: float


LOGGER = logging.getLogger(__name__)

DECIMAL_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)(?:\s+|[.)\-–—]\s*)(?P<title>.+)$"
)
KEYWORD_HEADING_RE = re.compile(
    r"^\s*(?P<keyword>(?:Chương|Chuong|Phần|Phan|Mục|Muc|Điều|Dieu|Khoản|Khoan|Điểm|Diem|Tiết|Tiet|Phụ\s+lục|Phu\s+luc|Chương|MỤC|CHƯƠNG|PHẦN|PHỤ LỤC|SECTION|ARTICLE))\b",
    re.IGNORECASE,
)
PAREN_HEADING_RE = re.compile(
    r"^\s*\((?P<token>(?:[ivxlcdm]+|[a-z]|[0-9]+))\)\s*(?P<title>.+)$",
    re.IGNORECASE,
)
ROMAN_HEADING_RE = re.compile(r"^\s*(?P<token>[IVXLCDM]+)(?:\.|\))\s*(?P<title>.+)$")
CAPTION_RE = re.compile(
    r"^\s*(Hình|Hinh|Bảng|Bang|Figure|Table|Biểu\s+đồ|Bieu\s+do|Sơ\s+đồ|So\s+do|Ảnh|Anh)\b",
    re.IGNORECASE,
)
TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d+$")
PAGE_NUMBER_RE = re.compile(r"^\s*(Trang|Page)\s*\d+(?:\s*/\s*\d+)?\s*$", re.IGNORECASE)
BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-–—•·●▪‣*]+)\s+")
REPEATED_TEXT_NORMALISER = re.compile(r"\s+")
SPLIT_TOKEN_PATTERN = re.compile(r"^[A-Za-zÀ-ỹĐđ]$")


class PdfDeepDocHTML(laws.Pdf):
    """DeepDoc variant that preserves layout hints for downstream HTML heuristics."""

    _logger = logging.getLogger(__name__)

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
        self._logger.info(
            "[LawsHTML] Dispatching to DeepDoc base parser for %s (pages %s-%s, zoom=%s)",
            filename,
            from_page,
            to_page,
            zoomin,
        )
        raw_sections, metadata = super().__call__(
            filename,
            binary=binary,
            from_page=from_page,
            to_page=to_page,
            zoomin=zoomin,
            callback=callback,
        )
        self._logger.debug(
            "[LawsHTML] DeepDoc returned %d raw sections", len(raw_sections)
        )
        boxes = list(getattr(self, "boxes", []))
        body_indents = [
            b.get("x0", 0.0)
            for b in boxes
            if self._is_body_candidate(b)
        ]
        body_indent = median(body_indents) if body_indents else 0.0

        sections: List[HtmlAwareSection] = []
        previous_bottom: dict[int, float] = {}
        for idx, (text, tags) in enumerate(raw_sections):
            clean_text = text.strip()
            if not clean_text:
                continue

            layout_type = ""
            level_hint: Optional[int] = None
            page = _extract_page_from_tags(tags)
            indent = 0.0
            font_size = 0.0
            height = 0.0
            gap_above: Optional[float] = None
            if idx < len(boxes):
                box = boxes[idx]
                layout_type = str(box.get("layout_type", ""))
                level_hint = self._estimate_heading_level(
                    box, clean_text, body_indent
                )
                indent = float(box.get("x0") or 0.0)
                top = float(box.get("top") or 0.0)
                bottom = float(box.get("bottom") or top)
                height = max(0.0, bottom - top)
                font_size = self._extract_font_size(box)
                if page is not None:
                    previous = previous_bottom.get(page)
                    if previous is not None:
                        gap_above = top - previous
                    previous_bottom[page] = bottom

            sections.append(
                HtmlAwareSection(
                    text=clean_text,
                    tags=tags or "",
                    layout_type=layout_type,
                    level_hint=level_hint,
                    page=page,
                    indent=indent,
                    font_size=font_size,
                    height=height,
                    gap_above=gap_above,
                )
            )

        _classify_vietnamese_headings(sections)

        heading_counts = Counter(
            str(section.level) for section in sections if section.level is not None
        )
        layout_counts = Counter(section.layout_type or "body" for section in sections)
        self._logger.info(
            "[LawsHTML] Produced %d sections", len(sections)
        )
        if heading_counts:
            self._logger.info(
                "[LawsHTML] Heading distribution: %s", dict(heading_counts)
            )
        else:
            self._logger.info("[LawsHTML] No heading levels detected")
        if layout_counts:
            self._logger.debug(
                "[LawsHTML] Layout distribution: %s", dict(layout_counts)
            )
        if metadata:
            self._logger.debug(
                "[LawsHTML] Metadata keys: %s", sorted(metadata.keys())
            )

        return sections, metadata

    @staticmethod
    def _extract_font_size(box: dict) -> float:
        size = box.get("font_size") or box.get("size")
        if size is None:
            font_data = box.get("font")
            if isinstance(font_data, dict):
                size = font_data.get("size")
        try:
            return float(size) if size is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

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


def _normalise_repetition(text: str) -> str:
    return REPEATED_TEXT_NORMALISER.sub(" ", text.strip().lower())


def _collect_body_stats(sections: Iterable[HtmlAwareSection]) -> BodyStats:
    font_values = [section.font_size for section in sections if section.font_size > 0]
    height_values = [section.height for section in sections if section.height > 0]
    gap_values = [
        section.gap_above for section in sections if section.gap_above is not None and section.gap_above > 0
    ]

    font_median = median(font_values) if font_values else 0.0
    height_median = median(height_values) if height_values else 0.0
    gap_median = median(gap_values) if gap_values else 0.0

    return BodyStats(font_median=font_median, height_median=height_median, gap_median=gap_median)


def _identify_repeating_lines(sections: Iterable[HtmlAwareSection]) -> set[str]:
    occurrences: dict[str, set[int]] = {}
    for section in sections:
        page = section.page
        if page is None:
            continue
        normalized = _normalise_repetition(section.text)
        if not normalized:
            continue
        occurrences.setdefault(normalized, set()).add(page)

    return {text for text, pages in occurrences.items() if len(pages) >= 2}


def _build_style_key(
    section: HtmlAwareSection,
    features: set[str],
    numeric_depth: Optional[int],
) -> str:
    tokens: list[str] = []
    core_features = [feature for feature in features if not feature.startswith("aux:")]
    tokens.extend(sorted(core_features))
    if numeric_depth:
        tokens.append(f"depth:{min(max(numeric_depth, 1), 6)}")
    indent_bucket = int(round(section.indent / 10.0)) if section.indent else 0
    tokens.append(f"indent:{indent_bucket}")
    if section.font_size:
        tokens.append(f"font:{int(round(section.font_size))}")
    return "|".join(tokens)


def _classify_heading_vi(
    index: int,
    section: HtmlAwareSection,
    repeated_lines: set[str],
    stats: BodyStats,
) -> Optional[HeadingCandidate]:
    text = section.text.strip()
    if not text:
        return None

    normalized = REPEATED_TEXT_NORMALISER.sub(" ", text)

    if normalized.lower() in repeated_lines:
        return None

    if CAPTION_RE.match(normalized):
        return None

    if TOC_LEADER_RE.search(normalized):
        return None

    if PAGE_NUMBER_RE.match(normalized):
        return None

    if BULLET_PREFIX_RE.match(normalized):
        return None

    if SPLIT_TOKEN_PATTERN.match(normalized):
        return None

    if normalized.count(" ") >= 12 and normalized.count(",") >= 2 and not normalized.endswith(":"):
        return None

    features: set[str] = set()
    numeric_depth: Optional[int] = None

    decimal_match = DECIMAL_HEADING_RE.match(normalized)
    if decimal_match:
        numeric_depth = len(decimal_match.group("number").split("."))
        features.add("decimal")

    keyword_match = KEYWORD_HEADING_RE.match(normalized)
    if keyword_match:
        keyword = _normalise_repetition(keyword_match.group("keyword"))
        features.add(f"keyword:{keyword}")
        if keyword in {"section", "article"} and not decimal_match:
            numeric_depth = numeric_depth or 2

    paren_match = PAREN_HEADING_RE.match(normalized)
    if paren_match:
        token = paren_match.group("token").lower()
        features.add("paren")
        if token.isdigit():
            numeric_depth = numeric_depth or 2
        elif re.fullmatch(r"[ivxlcdm]+", token):
            numeric_depth = numeric_depth or 2
        elif len(token) == 1 and token.isalpha():
            numeric_depth = numeric_depth or 3

    roman_match = ROMAN_HEADING_RE.match(normalized)
    if roman_match:
        features.add("roman")
        numeric_depth = numeric_depth or 2

    if normalized.isupper() and len(normalized.split()) <= 16:
        features.add("caps")

    if normalized.endswith(":") and len(normalized.split()) <= 20:
        features.add("colon")

    if section.level_hint is not None:
        features.add(f"hint:{section.level_hint}")

    gap = section.gap_above or 0.0
    if stats.gap_median and gap >= stats.gap_median * 1.2:
        features.add("aux:gap")

    font = section.font_size or 0.0
    if stats.font_median and font >= stats.font_median * 1.05:
        features.add("aux:font")

    height = section.height or 0.0
    if stats.height_median and height >= stats.height_median * 1.05:
        features.add("aux:height")

    if not features:
        return None

    if normalized.endswith(('.', '?', '!')) and "colon" not in features and "decimal" not in features:
        return None

    core_features = {feature for feature in features if not feature.startswith("aux:")}
    if not core_features:
        return None

    style_key = _build_style_key(section, features, numeric_depth)

    return HeadingCandidate(
        index=index,
        normalized=normalized,
        style_key=style_key,
        numeric_depth=numeric_depth,
        features=frozenset(features),
        page=section.page,
    )


def _assign_heading_levels(candidates: dict[int, HeadingCandidate]) -> None:
    style_order: dict[str, int] = {}
    next_level = 1
    for idx in sorted(candidates):
        candidate = candidates[idx]
        if candidate.style_key not in style_order:
            style_order[candidate.style_key] = next_level
            next_level = min(next_level + 1, 6)
        candidate.assigned_level = style_order[candidate.style_key]

    if len(style_order) == 1:
        for candidate in candidates.values():
            if candidate.numeric_depth:
                candidate.assigned_level = min(max(candidate.numeric_depth, 1), 6)


def _demote_false_titles(candidates: dict[int, HeadingCandidate]) -> None:
    for candidate in candidates.values():
        if candidate.assigned_level is None:
            continue
        words = candidate.normalized.split()
        if len(words) >= 35:
            candidate.assigned_level = None
            continue
        core_features = {feature for feature in candidate.features if not feature.startswith("aux:")}
        if not core_features:
            candidate.assigned_level = None
            continue
        if candidate.normalized.count(".") >= 4 and "decimal" not in core_features:
            candidate.assigned_level = None


def _audit_headings(candidates: dict[int, HeadingCandidate]) -> None:
    assigned = [cand.assigned_level for cand in candidates.values() if cand.assigned_level]
    if not assigned:
        LOGGER.info("[LawsHTML] Vietnamese heading audit: no headings classified")
        demoted_samples = [cand.normalized for cand in candidates.values()][:5]
        if demoted_samples:
            LOGGER.debug("[LawsHTML] Rejected heading samples: %s", demoted_samples)
        return

    distribution = Counter(assigned)
    LOGGER.info("[LawsHTML] Vietnamese heading audit distribution: %s", dict(distribution))
    demoted = [cand.normalized for cand in candidates.values() if not cand.assigned_level][:5]
    if demoted:
        LOGGER.debug("[LawsHTML] Demoted heading candidates: %s", demoted)


def _classify_vietnamese_headings(sections: List[HtmlAwareSection]) -> None:
    if not sections:
        return

    stats = _collect_body_stats(sections)
    repeated_lines = _identify_repeating_lines(sections)

    candidates: dict[int, HeadingCandidate] = {}
    for idx, section in enumerate(sections):
        candidate = _classify_heading_vi(idx, section, repeated_lines, stats)
        if candidate:
            candidates[idx] = candidate

    if candidates:
        _assign_heading_levels(candidates)
        _demote_false_titles(candidates)
        _audit_headings(candidates)
    else:
        LOGGER.info("[LawsHTML] Vietnamese heading audit: no heading candidates detected")

    for idx, section in enumerate(sections):
        candidate = candidates.get(idx)
        if candidate and candidate.assigned_level:
            section.level = min(candidate.assigned_level, 6)
        else:
            section.level = section.level_hint


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
        page = section.page if section.page is not None else _extract_page_from_tags(section.tags)
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
