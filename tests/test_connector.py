# tests/test_connector.py
import os
import shutil
from unittest.mock import patch
from pyutils.database.connector import manage_database


def test_backup_database(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    source.write_bytes(b"SQLite data")

    with patch('builtins.input', side_effect=['1', '3']):
        manage_database(str(source), str(backup))

    assert backup.exists()
    assert backup.read_bytes() == b"SQLite data"


def test_restore_database(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    backup.write_bytes(b"Backup data")
    source.write_bytes(b"Old data")

    with patch('builtins.input', side_effect=['2', '3']):
        manage_database(str(source), str(backup))

    assert source.read_bytes() == b"Backup data"
