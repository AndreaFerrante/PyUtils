# tests/test_localhost.py
import os
import pytest
from pyutils.service_factory.localhost import (
    get_files_in_path, get_file_in_root_path,
    get_file_content, get_files_timestamps_in_path
)


def test_get_files_in_path_all(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.csv").write_text("col1,col2")
    files = get_files_in_path(str(tmp_path))
    assert len(files) == 2


def test_get_files_in_path_filtered(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.csv").write_text("col1,col2")
    files = get_files_in_path(str(tmp_path), file_extension='txt')
    assert files == ['a.txt']


def test_get_file_in_root_path_finds_file(tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    target = subdir / "config.yaml"
    target.write_text("key: value")
    results = get_file_in_root_path(str(tmp_path), 'config.yaml')
    assert len(results) == 1
    assert 'config.yaml' in results[0]


def test_get_file_content(tmp_path):
    f = tmp_path / "query.sql"
    f.write_text("SELECT 1")
    content = get_file_content(str(tmp_path), 'query.sql')
    assert 'SELECT 1' in content


def test_get_files_timestamps_in_path_returns_sorted(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    result = get_files_timestamps_in_path(str(tmp_path))
    assert isinstance(result, list)
    assert len(result) == 2
    # Each entry is (filename, timestamp)
    assert isinstance(result[0], tuple)
