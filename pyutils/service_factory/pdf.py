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
