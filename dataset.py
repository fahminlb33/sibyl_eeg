import os
import csv
import argparse
import tempfile

from tqdm import tqdm
from termcolor import colored

from sibyl import dataset
from sibyl import transformer
from sibyl.util import filesystem as fs

parser = argparse.ArgumentParser(description="Download and transform EEG dataset")
parser.add_argument("action",
                    type=str,
                    choices=["download", "transform", "compress"],
                    help="Action can be download/transform/compress the dataset")
parser.add_argument("dataset",
                    type=str,
                    help="If the action is download, specifies the dataset size (small, large, full), otherwise specifies dataset path (.tar.gz)")
parser.add_argument("output-path",
                    type=str,
                    help="Output dataset path. The resulting file will be in .tar.gz if the action is download and .csv or parquet format if compression is enabled")

def download_dataset(dataset_type: str, output_path: str):
    print(colored("Starting to download {} dataset".format(dataset_type), "cyan"))

    url = dataset.get_download_url(dataset_type)
    print(colored("Downloading {}".format(url), "cyan"))

    dataset.async_download(url, output_path)
    print(colored("Download finished.", "cyan"))

def process_dataset(dataset_path: str, output_path: str):
    print(colored("Transforming dataset...", "cyan"))

    temp_dataset_path = tempfile.mkdtemp(prefix="sibyl_eeg_temp")

    # if the passed path is a file, extract it first
    print(colored("Decompressing main dataset archive...", "cyan"))
    fs.decompress_tar(dataset_path, temp_dataset_path)

    # decompress all tar files
    for data_file in tqdm(fs.find_files(temp_dataset_path, ".tar.gz"), desc="Decompressing files (step 1)"):
        fs.decompress_tar(data_file, temp_dataset_path)
        fs.delete_file(data_file)

    # decompress all gz files
    for data_file in tqdm(fs.find_files(temp_dataset_path, ".gz"), desc="Decompressing files (step 2)"):
        sample_extract_path = os.path.join(temp_dataset_path, os.path.basename(data_file) + ".txt")
        fs.decompress_gz(data_file, sample_extract_path)
        fs.delete_file(data_file)

    # process all files
    with open(output_path, 'w', newline='', encoding='utf-8') as file_stream:
        csv_writer = csv.writer(file_stream)
        transformer.write_csv_header(csv_writer)

        for record_file in tqdm(fs.find_files(temp_dataset_path, ".txt"), desc="Parsing dataset"):
            rows = transformer.parse_file(record_file)
            if rows is None:
                continue

            for row in rows:
                csv_writer.writerow(row)

            fs.delete_file(record_file)

    # delete temporary directory
    print(colored("Deleting temporary files...", "cyan"))
    fs.delete_dir(temp_dataset_path)

    print(colored("Transform complete!", "cyan"))

def compress_dataset(dataset_path: str, output_path: str):
    print(colored("Saving as parquet file with gzip compression, might take a while!", "cyan"))
    print(colored("Compressing...", "cyan"))

    transformer.save_as_parquet(dataset_path, output_path)

    print(colored("Saved as {}".format(output_path), "cyan"))


# main app entry point
if __name__ == "__main__":
    args = vars(parser.parse_args())

    # download dataset from UCI server (.tar.gz)
    if args["action"] == "download":
        if (args["dataset"] not in ["small", "large", "full"]):
            print(colored("Unknown dataset type, valid values are: small, large, full", "red"))
            exit()

        download_dataset(args["dataset"], args["output-path"])

    # transform dataset (.tar.gz to .csv)
    elif args["action"] == "transform":
        if not fs.is_file_exists(args["dataset"]):
            print(colored("Dataset file does not exists", "red"))
            exit()

        if not fs.is_file_extension(args["dataset"], [".tar.gz", ".tar"]):
            print(colored("Dataset is not in .tar or .tar.gz extension", "red"))
            exit()

        process_dataset(args["dataset"], args["output-path"])

    # compress dataset (.csv to .parquet)
    elif args["action"] == "compress":
        if not fs.is_file_exists(args["dataset"]):
            print(colored("Dataset file does not exists", "red"))
            exit()

        if not fs.is_file_extension(args["dataset"], [".csv"]):
            print(colored("Dataset is not in .csv extension", "red"))
            exit()

        compress_dataset(args["dataset"], args["output-path"])

    else:
        print("Unknown action, valid values are: download, transform")
