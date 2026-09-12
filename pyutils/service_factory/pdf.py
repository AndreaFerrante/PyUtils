import os

import pymupdf as fitz
import PyPDF2
from reportlab.pdfgen import canvas


def pdf_generator_from_text(output_file: str, pdf_text: str) -> None:
    """Generate a single-page PDF containing pdf_text at a fixed position."""
    c = canvas.Canvas(output_file)
    c.drawString(100, 750, pdf_text)
    c.save()


def scrape_pdf_content(pdf_path: str) -> str:
    """Extract and return all text from a PDF file.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    try:
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            return "".join(page.extract_text() or "" for page in reader.pages)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")


def pdf_pages_to_images(pdf_path: str, output_dir: str, dpi: int = 480) -> list[str]:
    """Render each PDF page to a PNG in output_dir, return the output paths in page order.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    paths = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out_path = os.path.join(output_dir, f"page_{i + 1}.png")
            page.get_pixmap(matrix=matrix).save(out_path)
            paths.append(out_path)
    return paths
