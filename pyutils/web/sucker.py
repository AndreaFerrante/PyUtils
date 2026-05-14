import smtplib
import http.server
import socketserver
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from bs4 import BeautifulSoup


def dummy_web_server(port: int = 8000) -> None:
    """Start a blocking HTTP server on port, serving files from the current directory."""
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()


def simple_scrape(url_to_scrape: str) -> str:
    """Fetch url_to_scrape and return the text of the first div.content element."""
    response = requests.get(url_to_scrape, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    element = soup.find("div", class_="content")
    if element is None:
        return ""
    return element.get_text()


def email_sender(
    smtp_server: str | None = None,
    sender_email: str | None = None,
    receiver_email: str | None = None,
    password: str | None = None,
    subject: str | None = None,
    body: str | None = None,
) -> None:
    """Send a plain-text email via SMTP (port 587 with STARTTLS).

    Returns immediately if any parameter is None.
    """
    params = {
        "smtp_server": smtp_server,
        "sender_email": sender_email,
        "receiver_email": receiver_email,
        "password": password,
        "subject": subject,
        "body": body,
    }
    for k, v in params.items():
        if v is None:
            print(f"Parameter {k} is None. Pass a string value")
            return

    try:
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_server, 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except (smtplib.SMTPException, OSError) as ex:
        print(f"email_sender error: {ex}")
