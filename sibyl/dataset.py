import os

from urllib import request
from urllib.parse import urlparse

from sibyl.util import DownloadProgressBar

def get_download_url(dataset_kind: str = "small"):
    if dataset_kind == "small":
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/smni_eeg_data.tar.gz"

    elif dataset_kind == "large":
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TRAIN.tar.gz"

    elif dataset_kind == "full":
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/eeg_full.tar"

    else:
        raise ValueError("dataset_kind must be one of small, large, or full")

def async_download(uri: str = None, save_path: str = None):
    file_name = uri.split('/')[-1]
    save_path = os.path.join(save_path, file_name)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:
        request.urlretrieve(uri, filename=save_path, reporthook=t.update_to)
