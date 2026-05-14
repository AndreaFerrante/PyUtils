import os
import pytest


@pytest.fixture
def temp_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("name,value\nalice,10\nbob,20\ncharlie,30\n")
    return str(csv_file)


@pytest.fixture
def temp_pdf(tmp_path):
    from reportlab.pdfgen import canvas
    pdf_path = str(tmp_path / "sample.pdf")
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, "Hello PyUtils test content")
    c.save()
    return pdf_path
