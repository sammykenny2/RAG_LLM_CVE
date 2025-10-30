"""
PDF processing module.
Handles text extraction from PDF files using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
import gc
from pathlib import Path
from typing import Union, Optional

# Import configuration
from config import (
    VERBOSE_LOGGING
)


class PDFProcessor:
    """
    PDF text extraction with optional memory optimization.

    Example usage:
        processor = PDFProcessor()
        text = processor.extract_text("report.pdf", max_pages=10)
    """

    def __init__(
        self,
        periodic_gc: bool = False,
        gc_frequency: int = 5
    ):
        """
        Initialize PDF processor.

        Args:
            periodic_gc: Enable periodic garbage collection during extraction
            gc_frequency: Run GC every N pages (if periodic_gc is True)
        """
        self.periodic_gc = periodic_gc
        self.gc_frequency = gc_frequency

    def extract_text(
        self,
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (None for all)
            max_length: Maximum text length (None for no limit)

        Returns:
            str: Extracted text
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if VERBOSE_LOGGING:
            print(f"Extracting text from: {pdf_path}")

        pdf_document = fitz.open(pdf_path)
        all_text = ""

        # Determine pages to process
        if max_pages is not None:
            pages_to_process = min(len(pdf_document), max_pages)
        else:
            pages_to_process = len(pdf_document)

        if VERBOSE_LOGGING:
            print(f"  └─ Processing {pages_to_process} pages")

        # Extract text from pages
        for page_num in range(pages_to_process):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            all_text += text

            # Periodic memory cleanup
            if self.periodic_gc and page_num % self.gc_frequency == 0:
                gc.collect()

            # Check length limit
            if max_length is not None and len(all_text) >= max_length:
                all_text = all_text[:max_length] + "..."
                break

        pdf_document.close()

        if VERBOSE_LOGGING:
            print(f"  └─ Extracted {len(all_text)} characters")

        return all_text

    def extract_text_by_pages(
        self,
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None
    ) -> list[dict]:
        """
        Extract text from PDF, returning list of page dicts.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (None for all)

        Returns:
            list: List of dicts with keys 'page_num', 'text'
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if VERBOSE_LOGGING:
            print(f"Extracting pages from: {pdf_path}")

        pdf_document = fitz.open(pdf_path)
        pages = []

        # Determine pages to process
        if max_pages is not None:
            pages_to_process = min(len(pdf_document), max_pages)
        else:
            pages_to_process = len(pdf_document)

        # Extract text from each page
        for page_num in range(pages_to_process):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            pages.append({
                'page_num': page_num + 1,  # 1-indexed
                'text': text
            })

            # Periodic memory cleanup
            if self.periodic_gc and page_num % self.gc_frequency == 0:
                gc.collect()

        pdf_document.close()

        if VERBOSE_LOGGING:
            print(f"  └─ Extracted {len(pages)} pages")

        return pages


def extract_cve_context(source_text: str, cve: str, window_chars: int = 1500) -> str:
    """
    Extract context window around CVE mentions in text.

    Args:
        source_text: Full text to search
        cve: CVE identifier (e.g., "CVE-2025-12345")
        window_chars: Size of context window (characters before + after)

    Returns:
        str: Extracted context snippets (or truncated text if no matches)
    """
    pattern = re.compile(re.escape(cve), re.IGNORECASE)
    snippets = []

    for match in pattern.finditer(source_text):
        half = window_chars // 2
        start = max(match.start() - half, 0)
        end = min(match.end() + half, len(source_text))
        snippets.append(source_text[start:end])

    if snippets:
        return "\n...\n".join(snippets)

    # No matches found, return truncated text
    return source_text[:window_chars]


# =============================================================================
# Utility functions (backward compatible)
# =============================================================================

def extract_text_from_pdf(
    pdf_name: Union[str, Path],
    max_pages: Optional[int] = None,
    periodic_gc: bool = False
) -> str:
    """
    Extract text from PDF (backward compatible function).

    Args:
        pdf_name: Path to PDF file
        max_pages: Maximum pages to process
        periodic_gc: Enable periodic garbage collection

    Returns:
        str: Extracted text
    """
    processor = PDFProcessor(periodic_gc=periodic_gc)
    return processor.extract_text(pdf_name, max_pages=max_pages)


# Import re for extract_cve_context
import re
