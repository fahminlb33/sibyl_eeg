import os
import csv
import gzip
import shutil
import tarfile

import pandas as pd

from sibyl.util import filesystem as fs

def write_csv_header(csv_writer):
    cols = ['id', 'trial', 'stimuli', 'sample', 'class', 'AF1', 'AF2', 'AF7', 'AF8',
       'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CP6', 'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
       'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FP1', 'FP2',
       'FPZ', 'FT7', 'FT8', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4',
       'P5', 'P6', 'P7', 'P8', 'PO1', 'PO2', 'PO7', 'PO8', 'POZ', 'PZ', 'T7',
       'T8', 'TP7', 'TP8', 'X', 'Y', 'nd']
    csv_writer.writerow(cols)

def parse_file(file_path: str):
    header1 = None
    header4 = None
    is_empty = True

    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    header1 = line
                elif i == 3:
                    header4 = line
                elif i == 10:
                    is_empty = False
                    break
    except:
        is_empty = True

    if is_empty:
        return None

    # read metadata
    col_id = header1[2:].strip()
    col_stimuli = header4.split(",")[0][1:].strip()
    col_trial = int(header4.split(",")[1][7:].strip())
    col_class = 1 if header1[5] == "a" else 0

    # transpose columns
    df = pd.read_csv(file_path, comment="#", sep=" ", names=["trial", "position", "sample", "value"])
    df_T = df.pivot(columns="position", index="sample", values="value").reset_index()
    df_T.insert(0, 'id', col_id)
    df_T.insert(1, 'trial', col_trial)
    df_T.insert(2, 'stimuli', col_stimuli)
    df_T.insert(4, 'class', col_class)

    return df_T.iloc[:, :].values

def save_as_parquet(file_path: str, output_path: str):
    df = pd.read_csv(file_path)
    df.to_parquet(output_path, compression="gzip")
