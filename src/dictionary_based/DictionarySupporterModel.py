import torch
import torch.nn as nn


def get_topk(self, out, target_index):
    target_val = out[target_index]
    return (out > target_val).sum().item()
        

        
class DictionarySupporterModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, quantize: bool = False):
        super(DictionarySupporterModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear_nn = nn.Linear(hidden_size, vocab_size)
        
        
        self.dense_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(), 
            nn.Linear(hidden_size, vocab_size),   
        )
                
        self.residual_nn = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )
    
        
        self.conv_nn = nn.Sequential(
            ConvBlock(hidden_size, hidden_size, kernel_size=3, activation_fn=nn.ReLU()),
            nn.Linear(hidden_size, vocab_size)
        )
    
        self.combined_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
    def forward_chunk(self, x):
        embedded = self.embedding(x)
        
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
        combined_out = self.combined_nn(embedded)
                
        return (linear_out, dense_out, residual_out, combined_out)

    def forward(self, x):
       
        embedded = self.embedding(x)
        
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
        combined_out = self.combined_nn(embedded)
        
        total = linear_out + dense_out + residual_out + combined_out
        
        return total



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
        out += residual
        return self.relu(out)


 # -------------  ---------------- #   
 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.activation = activation_fn

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, input_size, num_heads, num_layers):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


# Improves the model, but makes it twice as wlow.
class AttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, hidden_size)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        return x

class BatchNormBlock(nn.Module):
    def __init__(self, num_features):
        super(BatchNormBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batch_norm(x)

class DropoutBlock(nn.Module):
    def __init__(self, p):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)