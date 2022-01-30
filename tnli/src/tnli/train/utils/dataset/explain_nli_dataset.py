import torch
import numpy as np


class ExplainNLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        train_data, validation_data, test_data = load_explain_data(
            data_dir
        )
        self.result = train_data

    def __getitem__(self, item):
        return torch.LongTensor(self.result[0][item]), torch.LongTensor([self.result[2][item]]),\
               torch.LongTensor([self.result[1][item]])

    def __len__(self):
        return len(self.result[0])
        

def load_explain_data(data_dir):
    filepath = data_dir /  'ids_1.csv'
    with open(filepath, encoding='utf-8', mode='r') as f:
        input_ids = []
        line = f.readline()
        while line:
            if line != '\n':
                input_ids.append(list(map(int, line.replace("\n", "").split(','))))
            line = f.readline()
    input_ids = np.array(input_ids)
    filepath = data_dir / 'length_1.csv'
    lengths = np.loadtxt(filepath, delimiter=',')
    filepath = data_dir / 'labels_1.csv'
    labels = np.loadtxt(filepath, delimiter=',')

    return (input_ids, lengths, labels)
