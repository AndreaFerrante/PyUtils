import pytest
from pyutils.web.sucker import email_sender


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
