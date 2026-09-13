import os

import pytest
from pyutils.service_factory.pdf import scrape_pdf_content, pdf_generator_from_text, pdf_pages_to_images


def test_scrape_pdf_content_returns_text(temp_pdf):
    # Bug: PdfFileReader removed in PyPDF2 4.x → AttributeError
    text = scrape_pdf_content(temp_pdf)
    assert isinstance(text, str)
    assert len(text) > 0
    assert 'Hello' in text


def test_scrape_pdf_content_missing_file():
    with pytest.raises(FileNotFoundError):
        scrape_pdf_content('/nonexistent/path/file.pdf')


def test_pdf_generator_creates_file(tmp_path):
    output_path = str(tmp_path / "output.pdf")
    pdf_generator_from_text(output_path, "Test content")
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_pdf_pages_to_images_one_page_pdf(temp_pdf, tmp_path):
    out_dir = str(tmp_path / "pages")
    paths = pdf_pages_to_images(temp_pdf, out_dir)
    stem = os.path.splitext(os.path.basename(temp_pdf))[0]
    assert paths == [os.path.join(out_dir, f"{stem}_page_1.png")]
    assert os.path.getsize(paths[0]) > 0


def test_pdf_pages_to_images_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_pages_to_images('/nonexistent/path/file.pdf', str(tmp_path))
