import argparse

parser = argparse.ArgumentParser(description="Visualize EEG dataset")
parser.add_argument("action",
                    type=str,
                    choices=["channels", "signal", "signal_fft", "topograph", "topograph_animate"],
                    help="Action can be signal, signal_fft, topograph, topograph_animate")
parser.add_argument("dataset",
                    type=str,
                    help="Path to .csv or .parquet dataset file")
parser.add_argument("--id",
                    type=str,
                    help="EEG recording ID")
parser.add_argument("--trial",
                    type=int,
                    default=0,
                    help="EEG trial number")
parser.add_argument("--channel",
                    type=str,
                    default="AF1,F7",
                    help="EEG channel names, separated by comma (e.g. AF1,F7)")
parser.add_argument("--vspace",
                    type=int,
                    default=50,
                    help="Vertical spacing when plotting EEG recording")

# early parsing arguments
args = None
if __name__ == "__main__":
    # parse arguments
    args = vars(parser.parse_args())

# imports
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from termcolor import colored

from sibyl import plots
from sibyl.util import filesystem as fs

def visualize_channels(df: pd.DataFrame):
    print("Available channels:")
    print(df.columns.values[5:])

def visualize_signal(df: pd.DataFrame, vspace: int):
    values = df.values
    channels = df.columns.values

    _, ax = plt.subplots()
    plots.plot_eeg(values, channels, ax, vspace=vspace)
    plt.show()

def visualize_signal_fft(df: pd.DataFrame):
    _, ax = plt.subplots()
    plots.plot_eeg_fft(df, ax)
    plt.show()

def visualize_topograph(df: pd.DataFrame):
    ch_data = plots.reshape_data_topograph(df)
    pwrs, _ = plots.get_psds(ch_data)

    fig, ax = plt.subplots(figsize=(10,8))
    plots.plot_topomap(pwrs, ax, fig)
    plt.show()

def visualize_topograph_animate(df: pd.DataFrame):
    ch_data = plots.reshape_data_topograph(df)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10,8))

    chunk_data = np.array_split(ch_data, 10, axis=1)
    for chunk in chunk_data:
        pwrs, _ = plots.get_psds(chunk)
        ax.clear()
        plots.plot_topomap(pwrs, ax, fig, draw_cbar=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.1)

def query_subset(df: pd.DataFrame, trial: int, id: str):
    return df.loc[(df["trial"] == trial) & (df["id"] == id)]

def load_dataset(path: str):
    if ".csv" in path:
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)

# main app entry point
if __name__ == "__main__":
    # validations
    if not fs.is_file_extension(args["dataset"], [".csv", ".parquet"]):
        print(colored("Dataset is not in .csv or .parquet extension", "red"))
        exit()

    if not fs.is_file_exists(args["dataset"]) and not fs.is_directory_exists(args["dataset"]) :
        print(colored("Dataset file does not exists", "red"))
        exit()

    if args["id"] is None and args["action"] != "channels":
        print(colored("ID must be specified", "red"))
        exit()

    # load dataset
    df: pd.DataFrame = load_dataset(args["dataset"])

    # get available channels
    if args["action"] == "channels":
        visualize_channels(df)

    # EEG signal time series
    elif args["action"] == "signal":
        columns = args["channel"].split(",")
        chunk_df = query_subset(df, args["trial"], args["id"])[columns]

        visualize_signal(chunk_df, args["vspace"])

    # EEG signal time series with FFT filter
    elif args["action"] == "signal_fft":
        # validation
        if "," in args["channel"]:
            print(colored("FFT plot only available for a single channel", "red"))
            exit()

        chunk_df = query_subset(df, args["trial"], args["id"])[args["channel"]].values
        visualize_signal_fft(chunk_df)

    # EEG topograph
    elif args["action"] == "topograph":
        chunk_df = query_subset(df, args["trial"], args["id"])
        if len(chunk_df) == 0:
            print(colored("No data found using the specified query", "red"))
            exit()

        visualize_topograph(chunk_df)

    # EEG topograph with animation
    elif args["action"] == "topograph_animate":
        chunk_df = query_subset(df, args["trial"], args["id"])
        if len(chunk_df) == 0:
            print(colored("No data found using the specified query", "red"))
            exit()

        visualize_topograph_animate(chunk_df)

    # out of range
    else:
        print("Unknown action, valid values are: download, transform")
