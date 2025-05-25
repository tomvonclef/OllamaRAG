"""PDF utilities for OllamaRAG application."""
import fitz  # PyMuPDF
from typing import Tuple


def extract_text_from_pdf(file) -> Tuple[str, int]:
    """Extract all pages as text from an uploaded PDF file.

    Args:
        file: File-like object or path to the PDF file

    Returns:
        Tuple containing (extracted_text, page_count)

    Raises:
        ValueError: If the file is not a PDF or is empty
        RuntimeError: If the PDF is encrypted or corrupted
        IOError: If there are issues reading the file
    """
    # Check if file has a PDF extension
    if not file.name.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")

    try:
        # Read the file content once and keep it in memory
        file_content = file.read()
        if not file_content:
            raise ValueError("Uploaded file is empty")

        # Try to open the PDF
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
        except fitz.FileDataError as e:
            raise RuntimeError(
                "The file is not a valid PDF or is corrupted") from e
        except Exception as e:
            raise RuntimeError(f"Error opening PDF: {str(e)}") from e

        # Check if PDF is encrypted
        if doc.is_encrypted:
            # Try to decrypt with empty password (some PDFs use empty password)
            if not doc.authenticate(""):
                raise RuntimeError(
                    "PDF is encrypted and cannot be decrypted without a password")

        # Extract text from each page
        text = ""
        try:
            for page in doc:
                text += page.get_text()

            if not text.strip():
                raise RuntimeError(
                    "PDF appears to be empty or contains no extractable text")

            return text, len(doc)

        except Exception as e:
            raise RuntimeError(
                f"Error extracting text from PDF: {str(e)}") from e

        finally:
            # Ensure the document is always closed
            if 'doc' in locals():
                doc.close()

    except Exception as e:
        # Re-raise any unexpected errors with more context
        if not isinstance(e, (ValueError, RuntimeError)):
            raise RuntimeError(
                f"Unexpected error processing PDF: {str(e)}") from e
        raise
