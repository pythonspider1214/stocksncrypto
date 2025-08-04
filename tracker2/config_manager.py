
from dataclasses import dataclass
import json

@dataclass
class Config:
    api_key: str
    crypto_list: list
    stock_list: list

def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return Config(**data)
    except FileNotFoundError:
        print(f"Configuration file {file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file {file_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None
