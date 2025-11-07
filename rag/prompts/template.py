import os


PROMPT_DIR = os.path.dirname(__file__)

_loaded_prompts = {}

_VIETNAMESE_OUTPUT_NOTE = (
    "Chú thích sử dụng tiếng Việt: Khi phản hồi, hãy thêm một câu chú thích ngắn bằng "
    "tiếng Việt tóm tắt nội dung trả lời. Phần nội dung chính vẫn phải được viết bằng tiếng Anh."
)


def load_prompt(name: str) -> str:
    if name in _loaded_prompts:
        return _loaded_prompts[name]

    path = os.path.join(PROMPT_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file '{name}.md' not found in prompts/ directory.")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if _VIETNAMESE_OUTPUT_NOTE not in content:
            content = f"{content}\n\n{_VIETNAMESE_OUTPUT_NOTE}"

        _loaded_prompts[name] = content
        return content
