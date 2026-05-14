import pytest
import sys
import importlib.util

# Load sucker.py directly without going through __init__.py
spec = importlib.util.spec_from_file_location("sucker", "/Users/andrea/Documents/Github/PyUtils/pyutils/web/sucker.py")
sucker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sucker)
email_sender = sucker.email_sender


def test_email_sender_none_param_returns_early(capsys):
    # Bug: `for k, v in params:` raises ValueError (not enough values to unpack)
    # After fix: should print message and return None
    result = email_sender(
        smtp_server=None,
        sender_email='a@example.com',
        receiver_email='b@example.com',
        password='pass',
        subject='test',
        body='body'
    )
    assert result is None
    captured = capsys.readouterr()
    assert 'None' in captured.out or 'smtp_server' in captured.out


def test_email_sender_all_none_returns_early(capsys):
    result = email_sender()
    assert result is None
