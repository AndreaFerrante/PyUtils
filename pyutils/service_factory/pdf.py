import os

import pymupdf


def pdf_generator_from_text(output_file: str, pdf_text: str) -> None:
    """Generate a single A4 page PDF containing pdf_text at a fixed position."""
    with pymupdf.open() as doc:
        doc.new_page().insert_text((100, 92), pdf_text)
        doc.save(output_file)


def scrape_pdf_content(pdf_path: str) -> str:
    """Extract and return all text from a PDF file.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    with pymupdf.open(pdf_path) as doc:
        return "".join(page.get_text() for page in doc)


def pdf_pages_to_images(pdf_path: str, output_dir: str, dpi: int = 480) -> list[str]:
    """Render each PDF page to a PNG in output_dir, return the output paths in page order.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]

    paths = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out_path = os.path.join(output_dir, f"{stem}_page_{i + 1}.png")
            page.get_pixmap(dpi=dpi).save(out_path)
            paths.append(out_path)
    return paths
