import pytest
from pyutils.service_factory.pdf import scrape_pdf_content, pdf_generator_from_text


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
    import os
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
