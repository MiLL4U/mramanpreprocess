import csv
from typing import Dict, List


def csv_to_list(path: str, header: bool = True) -> List[str]:
    with open(path, 'r') as f:
        rows = list(csv.reader(f))
        if header:
            rows.pop(0)
        return [row[0] for row in rows]


def csv_to_2d_list(path: str, header: bool = True) -> List[List[str]]:
    with open(path, 'r') as f:
        rows = list(csv.reader(f))
        if header:
            rows.pop(0)
        return rows


def csv_to_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, 'r') as f:
        csv_rows = list(csv.reader(f))
        header = csv_rows.pop(0)
        res = [{key: value
                for key, value in zip(header, row)}
               for row in csv_rows]
    return res
