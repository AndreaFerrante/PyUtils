# tests/test_password.py
import string
from pyutils.web.password import generate_password


def test_generate_password_default_length():
    pwd = generate_password()
    assert len(pwd) == 12


def test_generate_password_custom_length():
    pwd = generate_password(length=20)
    assert len(pwd) == 20


def test_generate_password_valid_chars():
    valid = set(string.ascii_letters + string.digits + string.punctuation)
    pwd = generate_password(length=50)
    assert all(c in valid for c in pwd)


def test_generate_password_returns_string():
    pwd = generate_password()
    assert isinstance(pwd, str)
