def extract_keys_and_lists(d: dict, parent_key='', sep='_') -> list: 
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(extract_keys_and_lists(v, new_key, sep))
        elif isinstance(v, list):
            items.append((new_key, v))
    return items

def recursive_merge(dicts: list[dict]) -> dict:
    if not dicts:
        return {}

    merged = {}

    for key in dicts[0].keys():
        if isinstance(dicts[0][key], dict):
            sub_dicts = [d[key] for d in dicts if key in d]
            merged[key] = recursive_merge(sub_dicts)
        else:
            merged[key] = [d.get(key) for d in dicts]

    return merged

import time

class Timer:
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def end(self) -> None:
        self.end_time = time.perf_counter()

    def get_tot_time(self) -> tuple[int, int, int, float]:
        if self.start_time is None or self.end_time is None:
            return (-1, -1, -1, -1.0)
        
        tot_time = self.end_time - self.start_time
        
        return Timer.convert_time(tot_time)

    def get_average_time(self, divisor: int) -> tuple[int, int, int, float]:
        if divisor <= 0:
            raise ValueError("Divisor must be greater than 0.")
            
        if self.start_time is None or self.end_time is None:
            return (-1, -1, -1, -1.0)
        
        tot_time = self.end_time - self.start_time
        avg_time = tot_time / divisor

        return Timer.convert_time(avg_time)
    @staticmethod
    def convert_time(seconds: float) -> tuple[int, int, int, float]:
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(days), int(hours), int(minutes), seconds

import base64

def encode_to_base64(input_string: str) -> str:
    encoded_bytes = base64.b64encode(input_string.encode("utf-8"))
    encoded_string = encoded_bytes.decode("utf-8").rstrip("=")
    return encoded_string

def decode_from_base64(encoded_string: str) -> str:
    padding = 4 - (len(encoded_string) % 4)
    if padding != 4:
        encoded_string += "=" * padding
    decoded_bytes = base64.b64decode(encoded_string)
    decoded_string = decoded_bytes.decode("utf-8")
    return decoded_string
