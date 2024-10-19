
import torch.nn as nn

# IDEA: https://arxiv.org/pdf/1911.03572 (see Supporter model section)
class SupporterModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int):
        super(SupporterModel, self).__init__()
        # Embedding layer: Converts input indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Linear neural network: Quick learning of simple patterns
        self.linear_nn = nn.Linear(hidden_size, vocab_size)
        
        # Dense neural network: Moderate complexity for learning
        self.dense_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Residual neural network: Complex pattern learning
        self.residual_nn = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        # Convert input indices to embeddings
        embedded = self.embedding(x)
        
        # Process embeddings through each sub-network
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
        
        # Combine outputs from all sub-networks
        return linear_out + dense_out + residual_out

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