import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as text


class SummationEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, glove):
        super(SummationEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(len(glove), input_dim)
        self.embedding.weight = nn.Parameter(glove.vectors, requires_grad=True)
        self.dropout_input = nn.Dropout(dropout_rate)
        self.bottleneck = TanhLayer(input_dim, output_dim)
        self.droptout_output = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.embedding(x)
        h = torch.sum(h, 1)
        h = self.dropout_input(h)
        h = self.bottleneck(h)
        y = self.droptout_output(h)
        return y


class LSTMEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_rnn, dropout_rate, glove):
        super(LSTMEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(len(glove), input_dim)
        self.embedding.weight = nn.Parameter(glove.vectors, requires_grad=True)
        self.dropout_input = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_dim, input_dim, num_rnn, batch_first=True)
        self.bottleneck = TanhLayer(input_dim, output_dim)
        self.droptout_output = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.embedding(x)
        h = self.dropout_input(h)
        output, other = self.lstm(h)
        h = output[:, -1]
        print(h.shape)
        print(self.bottleneck)
        h = self.bottleneck(h)
        y = self.droptout_output(h)
        return y


class TanhLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TanhLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.linear(x)
        y = self.tanh(h)
        return y


class ClassificationLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        #self.softmax = nn.Softmax(dim=output_dim)

    def forward(self, x):
        h = self.linear(x)
        #y = self.softmax(h)
        return h


class OneLayerModel(nn.Module):
    def __init__(self, embedding_input_dim, embedding_output_dim, output_dim, dropout_rate, glove):
        super(OneLayerModel, self).__init__()
        self.embedding1 = SummationEmbeddingLayer(embedding_input_dim, embedding_output_dim, dropout_rate, glove)
        #self.embedding2 = SummationEmbeddingLayer(embedding_input_dim, embedding_output_dim, dropout_rate, glove)
        self.tanh1 = TanhLayer(embedding_output_dim * 2, embedding_output_dim * 2)
        self.tanh2 = TanhLayer(embedding_output_dim * 2, embedding_output_dim * 2)
        self.tanh3 = TanhLayer(embedding_output_dim * 2, embedding_output_dim * 2)
        self.classification_layer = ClassificationLayer(embedding_output_dim * 2, output_dim)

    def forward(self, x):
        print(x[0].shape, x[1].shape)
        h1 = self.embedding1(x[0])
        h2 = self.embedding1(x[1])
        print(h1.shape, h2.shape)
        h = torch.cat([h1, h2], dim=1)
        h = self.tanh1(h)
        h = self.tanh2(h)
        h = self.tanh3(h)
        y = self.classification_layer(h)
        return y


def test():
    from torchtext.vocab import GloVe
    glove = GloVe(name='42B', dim=300)
    model = OneLayerModel(embedding_input_dim=300, embedding_output_dim=100, output_dim=3, dropout_rate=0.5, glove=glove)
    examples = ['you', 'are', 'beautiful']
    example_ids = [glove.stoi[example] for example in examples]
    examples_2 = ['you', 'are', 'so', 'nice']
    example_ids_2 = [glove.stoi[example] for example in examples_2]
    example_tensor = torch.LongTensor([example_ids])
    example_tensor2 = torch.LongTensor([example_ids_2])
    input_example = [example_tensor, example_tensor2]
    model(input_example)
    print(model)


if __name__ == '__main__':
    test()