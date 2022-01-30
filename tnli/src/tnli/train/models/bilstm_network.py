import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        """
        LSTM
        :param embedding_size:
        :param hidden_size:
        :param out_size:
        """
        super(BiLSTMNetwork, self).__init__()
        self.fc1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.dropout1 = nn.Dropout(p=0.75)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.75)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=0.75)
        self.fc4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = F.softmax(out)
        return out
