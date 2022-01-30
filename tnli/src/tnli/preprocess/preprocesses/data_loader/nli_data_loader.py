from typing import Tuple

import numpy as np


class NliDataLoader:
    @staticmethod
    def load(input_filename: str)-> Tuple[np.ndarray]:
        sentence1s = []
        sentence2s = []
        labels = []
        with open(input_filename, encoding='utf-8', mode='r') as f:
            lines = f.readlines()
            l2n = {'contradiction':0, 'entailment':1, 'neutral':2}
            cnt = 0
            for line in lines[1:]:
                d = line.split("\t")
                if cnt % 100000 == 0:
                    print(cnt)
                if d[0] not in l2n.keys():
                    continue
                sentence1s.append(d[5])
                sentence2s.append(d[6])
                labels.append(l2n[d[0]])
                cnt += 1
        return labels, sentence1s, sentence2s