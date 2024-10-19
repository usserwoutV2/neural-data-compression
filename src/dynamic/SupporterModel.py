
import torch.nn as nn


class SupporterModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int):
        super(SupporterModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear_nn = nn.Linear(hidden_size, vocab_size)
        self.dense_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        self.residual_nn = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
        return linear_out + dense_out + residual_out

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += residual
        return self.relu(out)
      