from abc import ABC, abstractmethod
from pathlib import Path
import os
from typing import Any

import numpy as np

from .data_loader import DataLoader


class Preprocessor(ABC):
    @classmethod
    def process(
        cls, 
        encoder,
        input_dir_name: Path,
        output_dir_name: Path, 
        model_name: str,
        max_len: int,
        data_loader
    ) -> None:
        """
        Pre-process files in given directory
        outputs processed data

        Parameters
        ----------
        input_dir_name: Path
            the input directory name
            this must contain appropriate dataset
        output_dir_name: Path
            the output directory name
        model_name: str
            the model name for which this preprocess is done.
        """
        count = 1
        for input_filename in (input_dir_name).iterdir():
            labels, sentence1s, sentence2s = \
                data_loader.load(input_filename)

            data = \
                cls.encode(
                    (labels, sentence1s, sentence2s), 
                    encoder, max_len=max_len
                )

            os.makedirs(output_dir_name, exist_ok=True)

            cls.write(
                data,
                output_dir_name,
                model_name,
                count
            )
            count += 1
    
    @classmethod
    @abstractmethod
    def write(
        cls,
        data: list,
        output_dir_name: Path, 
        model_name: str,
        count: int):
        """
        write and save data
        """
        pass

    @classmethod
    @abstractmethod
    def encode(
        cls,
        data: tuple, 
        encoder: Any, 
        max_len: int):
        """
        encode a set of data
        """
        pass
