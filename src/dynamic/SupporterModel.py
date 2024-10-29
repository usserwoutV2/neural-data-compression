import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# IDEA: https://arxiv.org/pdf/1911.03572 (see Supporter model section)
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # We use this to reduce the precision of the input tensor for smaller model size
        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        embedded = self.embedding(x)
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
        x = linear_out + dense_out + residual_out
        x = self.dequant(x)
        return x

    def forward(self, x):
        # Convert input indices to embeddings
        embedded = self.embedding(x)
        
        # Process embeddings through each sub-network
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
                
        # Combine outputs from all sub-networks
        
        return  self.dequant(linear_out +  dense_out + residual_out)


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Store input for residual connection
        residual = x
        # Process input through linear layers and ReLU
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        # Add residual connection
        out += residual
        return self.relu(out)