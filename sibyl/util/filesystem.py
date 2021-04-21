import os
import gzip
import shutil
import tarfile

from pathlib import Path

from typing import List

def find_files(root: str, extension: str=None):
    found_files = []
    for path, _, files in os.walk(root):
        for name in files:
            if extension is not None and extension in name:
                found_files.append(os.path.join(path, name))

    return found_files

def delete_dir(path: str):
    try:
        shutil.rmtree(path)
    except OSError:
        #print ("Error: %s - %s." % (e.filename, e.strerror))
        pass

def delete_file(path: str):
    try:
        os.remove(path)
    except OSError:
        #print ("Error: %s - %s." % (e.filename, e.strerror))
        pass

def is_file_exists(path: str):
    return os.path.isfile(path)

def is_file_extension(path: str, extensions: List[str]):
    return len([ext in path for ext in extensions]) > 0

def decompress_tar(file_path: str, output_path: str):
    open_mode = "r:"
    if ".gz" in file_path:
        open_mode = "r:gz"

    with tarfile.open(file_path, open_mode) as tar_file:
        tar_file.extractall(output_path)

def decompress_gz(file_path: str, output_path: str):
    with gzip.open(file_path, "rb") as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
