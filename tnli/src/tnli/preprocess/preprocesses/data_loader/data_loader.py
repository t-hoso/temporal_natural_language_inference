from typing import Tuple

import numpy as np


class DataLoader:
    @staticmethod
    def load(input_filename: str)-> Tuple[np.ndarray]:
        raw_data = np.loadtxt(input_filename, delimiter="\t", dtype="str")
        data_body = raw_data[1:]
        return raw_data[1:][:,0], raw_data[1:][:,1], raw_data[1:][:,2]
