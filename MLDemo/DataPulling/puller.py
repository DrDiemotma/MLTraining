import os.path

import numpy as np
import pandas as pd
import requests
import tarfile
import urllib.request
import queue
import pickle

from sklearn.datasets import fetch_openml

from MLDemo.DataPulling import constants

from pathlib import Path


def pull_text_data(file: str, root: str = constants.DATA_ROOT, target_file: str | None = None) -> str | None:
    if not root.endswith("/"):
        root += "/"

    if target_file is None or len(target_file) == 0:
        tokens = file.split('/')
        target_file = tokens[len(tokens)-1]

    url: str = root + file

    try:
        request = requests.get(url)
        with open(target_file, "wb") as f:
            f.write(request.content)
    except Exception as e:
        print(f"Could not load file: {e}")
        return None

    return target_file

def pull_tgz_data(file: str, root: str = constants.DATA_ROOT, target_directory: str | None = None) -> str | None:
    if not root.endswith("/"):
        root += "/"

    if target_directory is None or len(target_directory) == 0:
        tokens = file.split('/')
        target_directory = tokens[len(tokens) - 1]
        tokens = target_directory.split(".")
        target_directory = tokens[0]

    if not os.path.isdir(target_directory):
        os.mkdir(target_directory)

    temp_index = 0
    while os.path.isfile(f"temp_file_{temp_index:04d}.tgz"):
        temp_index += 1

    temp_file = f"temp_file_{temp_index:04d}.tgz"

    url = root + file
    urllib.request.urlretrieve(url, temp_file)
    with tarfile.open(temp_file) as tarball:
        tarball.extractall(path=target_directory)

    os.remove(temp_file)
    return target_directory


def open_csv(file: str, stored_file: str | None = None, root: str = constants.DATA_ROOT, overwrite: bool = False,
             **kwargs) \
        -> pd.DataFrame | None:
    if stored_file is None:
        tokens = file.split('/')
        stored_file = tokens[len(tokens)-1]

    if not os.path.isfile(stored_file) or overwrite:
        loaded_file: str = pull_text_data(file, root, stored_file)
        if loaded_file is None:
            return None

    data = pd.read_csv(stored_file, **kwargs)
    return data

def open_tgz(file: str, directory: str | None = None, root: str = constants.DATA_ROOT, overwrite: bool = False) \
    -> list[tuple[str, pd.DataFrame]] | None:
    if directory is None or len(directory) == 0:
        tokens = file.split('/')
        directory = tokens[len(tokens)-1]
        tokens = directory.split('.')
        directory = tokens[0]

    def delete_recursively(local_directory):
        sub_dirs = []
        for loop_f in os.listdir(local_directory):
            if loop_f == "." or loop_f == "..":
                continue

            if os.path.isdir(os.path.join(local_directory, loop_f)):
                sub_dirs.append(os.path.join(local_directory, loop_f))
                continue

            os.remove(os.path.join(local_directory, loop_f))

        for loop_d in sub_dirs:
            delete_recursively(loop_d)

    if not os.path.isdir(directory):
        os.mkdir(directory)
        directory = pull_tgz_data(file, root, directory)
    elif overwrite:
        for f in os.listdir(directory):
            delete_recursively(os.path.join(directory, f))
        directory = pull_tgz_data(file, root, directory)

    if directory is None:
        return None

    data_files = []
    directories = queue.Queue()
    directories.put(directory)
    while not directories.empty():
        current_directory = directories.get()
        for f in os.listdir(current_directory):
            if os.path.isdir(f):
                directories.put(os.path.join(current_directory, f))
            elif f.endswith(".csv"):
                data_files.append(os.path.join(current_directory, f))

    def get_tuple(local_file: str) -> tuple[str, pd.DataFrame]:
        if not local_file.endswith(".csv"):
            raise NotImplemented(f"Format for file not implemented: {local_file}")

        local_tokens = local_file.split('/')
        local_name = local_tokens[len(local_tokens)-1]
        local_tokens = local_name.split('.')
        local_name: str = local_tokens[0]
        local_data_frame: pd.DataFrame = pd.read_csv(local_file)
        return local_name, local_data_frame

    output_tuples = [get_tuple(f) for f in data_files]

    return output_tuples


def get_mnist(file: str = "mnist.pkl", overwrite: bool = False) -> tuple[np.array, np.array]:
    """
    Get the MNIST data set.
    :param file: Locally stored file. Used so it doesn't need to be downloaded every time. Is a pickle file.
    :param overwrite: If the  file needs to be downloaded again even if the local file is present.
    :return: Tuple of Numpy arrays (data, label)
    """
    try:
        x: np.array = None
        y: np.array = None
        if overwrite or not os.path.isfile(file):
            mnist = fetch_openml('mnist_784', as_frame=False)
            x, y = mnist.data, mnist.target
            d = {"x": x, "y": y}
            with open(file, "wb") as f:
                pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        else:
            with open(file, "rb") as f:
                d = pickle.load(f)
                x, y = d["x"], d["y"]
    except (FileNotFoundError, OSError) as e:
        raise e
    except Exception as e:
        print(f"Unknown exception: {e}")
        raise e
    finally:
        return x, y


def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (("easy_ham", "ham", ham_url),
                                    ("spam", "spam", spam_url)):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            tar_bz2_file = tarfile.open(path)
            tar_bz2_file.extractall(path=spam_path)
            tar_bz2_file.close()
    return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]


if __name__ == "__main__":
    get_mnist()
