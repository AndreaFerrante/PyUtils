import os
import time
import zipfile
from pathlib import Path


def get_files_in_path(path_to_search: str, file_extension: str | None = None) -> list[str]:
    """Return filenames in path_to_search, optionally filtered by file_extension (without dot)."""
    files = [f.name for f in Path(path_to_search).iterdir() if f.is_file()]
    if file_extension:
        files = [f for f in files if Path(f).suffix.lstrip(".") == file_extension]
    return files


def get_file_in_root_path(path_to_search: str, file_name: str) -> list[str]:
    """Return all absolute paths matching file_name found recursively under path_to_search."""
    results = []
    for root, _, files in os.walk(path_to_search):
        if file_name in files:
            results.append(os.path.join(root, file_name))
    return results


def get_file_content(path_to_read: str, file_name: str) -> str:
    """Return the text content of path_to_read/file_name with newlines replaced by spaces."""
    with open(Path(path_to_read) / file_name) as fh:
        return fh.read().replace("\n", " ")


def get_files_timestamps_in_path(path_to_scan: str) -> list[tuple[str, str]]:
    """Return (filename, ctime) tuples for all files in path_to_scan, sorted newest first."""
    entries = [
        (p.name, p.stat().st_ctime)
        for p in Path(path_to_scan).iterdir()
        if p.is_file()
    ]
    return [(name, time.ctime(ts)) for name, ts in sorted(entries, key=lambda x: x[1], reverse=True)]


def get_files_zipped_in_folder(path_to_zip: str, file_extensions: list[str]) -> None:
    """Zip all files with matching extensions in path_to_zip, one archive per file."""
    base = Path(path_to_zip)
    for file_path in base.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lstrip(".") not in file_extensions:
            continue
        archive_path = base / (file_path.stem + ".zip")
        print(f"Compressing file named {file_path.name} . . .")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path, arcname=file_path.name)


def get_csv_files_in_path_stacked(
    path_to_stack: str,
    file_extensions: str = "csv",
    sep: str = ";",
    encoding: str = "cp1252",
    print_files: bool = True,
):
    """Return all CSV files in path_to_stack concatenated into a single DataFrame."""
    import pandas as pd

    frames = []
    for file_path in Path(path_to_stack).iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lstrip(".") not in file_extensions:
            continue
        if print_files:
            print(f"Reading file named {file_path.name} . . .")
        frames.append(pd.read_csv(file_path, sep=sep, encoding=encoding))
    return pd.concat(frames)


def get_xlsx_files_in_path_stacked(
    path_to_stack: str,
    file_extensions: list[str] | None = None,
    print_files: bool = True,
):
    """Return all Excel files in path_to_stack concatenated into a single DataFrame."""
    import pandas as pd

    if file_extensions is None:
        file_extensions = ["xls", "xlsx", "xlsm"]

    frames = []
    for file_path in Path(path_to_stack).iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lstrip(".") not in file_extensions:
            continue
        if print_files:
            print(f"Reading file named {file_path.name} . . .")
        frames.append(pd.read_excel(file_path))
    return pd.concat(frames)


def list_processes() -> list[dict]:
    """Return a list of dicts with pid and name for all running processes."""
    import psutil

    return [proc.info for proc in psutil.process_iter(["pid", "name"])]


def close_app(app_name: str) -> None:
    """Terminate all processes whose name contains app_name (case-insensitive)."""
    import psutil

    for proc in psutil.process_iter(["pid", "name"]):
        if app_name.lower() in (proc.info["name"] or "").lower():
            print(f"Closing app named {proc.info['name']}")
            psutil.Process(proc.info["pid"]).terminate()
