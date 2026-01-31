import json
import os
import pickle
import torch
import numpy as np
import random
import yaml

def read_json_file(file_path):
    """
    Read a JSON file and return its contents as a Python dictionary.

    :param file_path: The path to the JSON file.
    :type file_path: str
    :return: A dictionary representing the JSON data.
    :rtype: dict
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while reading the file {file_path}: {e}")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_pickle_file(file_path):
    """
    Read a pickle file and return its contents as a Python object.

    :param file_path: The path to the pickle file.
    :type file_path: str
    :return: The Python object stored in the pickle file.
    :rtype: object
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pickle.UnpicklingError as e:
        print(f"Error unpickling file {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while reading the file {file_path}: {e}")


def save_to_json(path,data):
        with open(path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=1)
        print("file have been successfully saved to\n", path)


def save_to_pickle(save_path, data):
    try:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        print("file have been successfully saved to", save_path)
    except IOError:
        print("Error: Unable to write to the file", save_path)
    except Exception as e:
        print("An error occurred while saving the file:", str(e))
        
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # si vous utilisez multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

def seed_everything(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Rend (en grande partie) les résultats reproductibles, parfois un peu plus lent.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Pour PyTorch récents: force des algos déterministes (peut lever une erreur si op non déterministe)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        # Optionnel (recommandé sur certains GPU pour matmul déterministe)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

