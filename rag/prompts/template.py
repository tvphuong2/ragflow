import os


PROMPT_DIR = os.path.dirname(__file__)

_loaded_prompts = {}

_VIETNAMESE_NOTE = (
    "\n\nChú ý (Tiếng Việt): Khi phản hồi, hãy bổ sung chú thích hoặc giải thích ngắn gọn "
    "bằng tiếng Việt để hỗ trợ người dùng, nhưng vẫn đảm bảo nội dung chính và định dạng "
    "được yêu cầu trình bày bằng tiếng Anh."
)


def load_prompt(name: str) -> str:
    if name in _loaded_prompts:
        return _loaded_prompts[name]

    path = os.path.join(PROMPT_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file '{name}.md' not found in prompts/ directory.")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if _VIETNAMESE_NOTE.strip() not in content:
            content = f"{content}{_VIETNAMESE_NOTE}"
        _loaded_prompts[name] = content
        return content
