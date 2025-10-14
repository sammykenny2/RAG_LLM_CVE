"""Remove all cached Llama 3.2 model artifacts from the local Hugging Face cache."""

import os
import shutil
from pathlib import Path


def candidate_roots() -> list[Path]:
    """Return likely cache roots, deduplicated."""
    seen = set()
    roots = []

    hints = [
        os.environ.get("HF_HOME"),
        os.environ.get("HF_HUB_CACHE"),
    ]

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:  # pragma: no cover
        HF_HUB_CACHE = None
    hints.append(HF_HUB_CACHE)

    defaults = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "huggingface" / "hub",
    ]

    for raw in hints + defaults:
        if not raw:
            continue
        path = Path(raw)
        if path in seen:
            continue
        seen.add(path)
        roots.append(path)

    return roots


def remove_dir(path: Path) -> bool:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def main() -> None:
    removed = []
    for root in candidate_roots():
        if not root.exists():
            continue
        # Find all Llama 3.2 model directories
        for item in root.iterdir():
            if item.is_dir() and "Llama-3.2" in item.name and "Instruct" in item.name:
                if remove_dir(item):
                    removed.append(item)
    if removed:
        print("Removed cached Llama 3.2 model directories:")
        for path in removed:
            print(f" - {path}")
    else:
        print("No cached Llama 3.2 model directories found.")


if __name__ == "__main__":
    main()
