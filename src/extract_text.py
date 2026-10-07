
from pathlib import Path

def read_txt_resume(path: str) -> str:
    """Read a plain-text resume. Extend this module for PDF/DOCX parsing."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")
