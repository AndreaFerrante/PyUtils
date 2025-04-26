
def pdf_generator_from_text(output_file:str, pdf_text:str) -> None:

    from reportlab.pdfgen import canvas

    """
     Generate a PDF file with given text.

     Parameters:
     - output_file (str): The file path where the generated PDF should be saved.
     - pdf_text (str): The text to be included in the PDF.

     Returns:
     - None
     """

    try:
        c = canvas.Canvas(output_file)
        c.drawString(100, 750, pdf_text)
        c.save()

    except Exception as ex:
        print(f'Function pdf_generator give error {ex}')

def scrape_pdf_content(pdf_path: str) -> str:

    import PyPDF2

    """
    Scrape all the text content from a PDF file.

    This function reads a PDF file from the given file path and extracts all the text content.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A string containing all the text content from the PDF.

    Raises:
        FileNotFoundError: Raised if the specified PDF file path does not exist.
        Exception: Any exception raised by the PyPDF2 library.
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            num_pages = pdf_reader.numPages

            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.getPage(page_num)
                full_text += page.extract_text()

        return full_text

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {pdf_path} was not found.")
    except Exception as ex:
        raise Exception(f"An error occurred while processing the PDF: {ex}")
