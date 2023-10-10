import requests
import smtplib
import http.server
import socketserver
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup


def dummy_web_server(port:int=8000):

    """
     Create a simple HTTP web server.

     This function sets up and starts a basic HTTP web server using Python's built-in
     `http.server.SimpleHTTPRequestHandler` and `socketserver.TCPServer`. The server will
     serve files from the current directory and below, automatically mapping the file
     structure to HTTP requests.

     Parameters:
     - port (int, optional): The port number on which the web server will listen.
                              Default is 8000.

     Notes:
     - The server will run indefinitely until manually terminated.
     - This is a simple server for demonstration or debugging purposes and is not
       suitable for production use.

     Example:
     >>> dummy_web_server(8080)
     Serving at port 8080
     """

    with socketserver.TCPServer(('', port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()


def simple_scrape(url_to_scrape:str=''):

    """
    Scrape and print the text content of a specified div element from a webpage.

    Parameters:
    - url_to_scrape (str): The URL of the webpage to scrape. Defaults to an empty string.

    Returns:
    - str: The text content of the div with class 'content'. Returns an empty string if the function fails to scrape.
    """

    try:
        response = requests.get(url_to_scrape)
        soup     = BeautifulSoup(response.text, 'html.parser')
        data     = soup.find('div', class_='content')
        print(data.text)
        return data.text

    except Exception as ex:
        print(f'Function simple_scrape gave error {ex}')


def email_sender(smtp_server:str=None,
                 sender_email:str=None,
                 receiver_email:str=None,
                 password:str=None,
                 subject:str=None,
                 body:str=None):

    """
    Sends an email using the Simple Mail Transfer Protocol (SMTP).

    This function constructs an email message with the given details and
    transmits it through the specified SMTP server. The function will
    return immediately and print a message if any required parameter is
    set to None.

    Args:
        smtp_server (str, optional): The hostname or IP address of the SMTP server. Default is None.
        sender_email (str, optional): The email address from which the email will be sent. Default is None.
        receiver_email (str, optional): The email address to which the email will be sent. Default is None.
        password (str, optional): Password for the sender's email account. Default is None.
        subject (str, optional): Subject line for the email. Default is None.
        body (str, optional): The main body text of the email. Default is None.

    Returns:
        None: Function returns None if any parameter is set to None or if an exception occurs.

    Raises:
        Exception: Any exception raised by the smtplib library or email.mime modules.
    """


    params = locals()
    for k, v in params:
        if v is None:
            print(f'Parameter {k} is None. Pass a string value')
            return

    try:
        message            = MIMEMultipart()
        message['From']    = sender_email
        message['To']      = receiver_email
        message['Subject'] = subject

        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(smtp_server, 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())

    except Exception as ex:
        print(f'Function email_sender generated error {ex}')