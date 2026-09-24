import pytest


@pytest.fixture
def temp_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("name,value\nalice,10\nbob,20\ncharlie,30\n")
    return str(csv_file)


@pytest.fixture
def temp_pdf(tmp_path):
    from pyutils.service_factory.pdf import pdf_generator_from_text
    pdf_path = str(tmp_path / "sample.pdf")
    pdf_generator_from_text(pdf_path, "Hello PyUtils test content")
    return pdf_path
