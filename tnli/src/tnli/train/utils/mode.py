from enum import Enum


class Mode(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3
    MISMATCHED = 4
    MATCHED = 5