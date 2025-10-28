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

"""Utilities for exporting LLM interaction logs to a dedicated file."""

from __future__ import annotations

import json
import os
from datetime import datetime
from threading import Lock
from typing import Any

_LOG_PATH_ENV = "LLM_INTERACTION_LOG_PATH"
_DEFAULT_LOG_PATH = os.path.join("logs", "llm", "llm_interactions.log")
_WRITE_LOCK = Lock()


def _safe_serialize(value: Any) -> Any:
    """Safely convert values to JSON-serialisable structures."""

    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]

    if isinstance(value, tuple):
        return [_safe_serialize(item) for item in value]

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:  # pragma: no cover - extremely rare
            return value.hex()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _assemble_prompt_text(system: str | None, history: list[dict[str, Any]]) -> str:
    """Create a human-readable prompt trace from system + history messages."""

    parts: list[str] = []
    if system:
        parts.append(f"system: {system}")

    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content")
        safe_content = _safe_serialize(content)
        if isinstance(safe_content, (dict, list)):
            content_repr = json.dumps(safe_content, ensure_ascii=False)
        else:
            content_repr = str(safe_content)
        parts.append(f"{role}: {content_repr}")

    return "\n".join(parts)


def export_llm_interaction_log(
    *,
    tenant_id: str | None,
    llm_name: str | None,
    model_name: str | None,
    provider: str | list[str] | None,
    mode: str,
    system_prompt: str | None,
    history: list[dict[str, Any]] | None,
    request: dict[str, Any] | None,
    output: Any,
    usage: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Export a single LLM interaction entry to the configured log file."""

    history = history or []
    sanitized_history = _safe_serialize(history)
    sanitized_request = _safe_serialize(request or {})
    sanitized_output = _safe_serialize(output)
    sanitized_usage = _safe_serialize(usage or {})
    sanitized_extra = _safe_serialize(extra or {})
    sanitized_provider = _safe_serialize(provider)

    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "tenant_id": tenant_id,
        "llm_name": llm_name,
        "model_name": model_name,
        "provider": sanitized_provider,
        "mode": mode,
        "full_prompt": {
            "system": system_prompt,
            "messages": sanitized_history,
            "text": _assemble_prompt_text(system_prompt, history),
        },
        "request": sanitized_request,
        "history": sanitized_history,
        "output": sanitized_output,
    }

    if sanitized_usage:
        record["usage"] = sanitized_usage

    if sanitized_extra:
        record["extra"] = sanitized_extra

    log_path = os.environ.get(_LOG_PATH_ENV, _DEFAULT_LOG_PATH)
    directory = os.path.dirname(log_path) or "."
    os.makedirs(directory, exist_ok=True)

    serialized = json.dumps(record, ensure_ascii=False)

    with _WRITE_LOCK:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(serialized + "\n")

