from glob import glob
from typing import Dict

import ibwpy as ip


def load_ibws(dir_path: str) -> Dict[str, ip.BinaryWave5]:
    ibw_paths = glob(dir_path + "*.ibw")

    res: Dict[str, ip.BinaryWave5] = {}
    for path in ibw_paths:
        ibw = ip.load(path)
        res[ibw.name] = ibw

    return res
