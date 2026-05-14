# tests/test_wol.py
from unittest.mock import patch, MagicMock
from pyutils.web.wol import send_magic_packet


def test_send_magic_packet_valid_mac(capsys):
    with patch('socket.socket') as mock_socket:
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock
        mock_sock.sendto.return_value = 102

        send_magic_packet('AA:BB:CC:DD:EE:FF')

        captured = capsys.readouterr()
        assert 'successfully' in captured.out.lower() or 'sent' in captured.out.lower()


def test_send_magic_packet_invalid_mac(capsys):
    # Invalid hex should be caught and printed
    send_magic_packet('ZZ:ZZ:ZZ:ZZ:ZZ:ZZ')
    captured = capsys.readouterr()
    assert 'failed' in captured.out.lower() or 'error' in captured.out.lower()
