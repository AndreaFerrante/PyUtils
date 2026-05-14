import pytest
from pyutils.data.analysis import csv_analyzer


def test_csv_analyzer_reads_given_path(temp_csv, capsys):
    # Must read the file at temp_csv path, not hardcoded 'data.csv'
    csv_analyzer(temp_csv, separator=',')
    captured = capsys.readouterr()
    # describe() output contains 'count' for numeric columns
    assert 'count' in captured.out


def test_csv_analyzer_single_column_mean(temp_csv, capsys):
    csv_analyzer(temp_csv, single_column='value', separator=',')
    captured = capsys.readouterr()
    assert 'Mean of value is: 20.0' in captured.out


def test_csv_analyzer_missing_file_prints_error(capsys):
    csv_analyzer('/nonexistent/path/data.csv', separator=',')
    captured = capsys.readouterr()
    assert 'error' in captured.out.lower()
