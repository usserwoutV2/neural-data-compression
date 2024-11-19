import torch
import torch.nn as nn

# IDEA: https://arxiv.org/pdf/1911.03572 (see Supporter model section)
class SupporterModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, quantize: bool = False):
        super(SupporterModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear_nn = nn.Linear(hidden_size, hidden_size)
        self.dense_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),  # Appears to perform better than ReLU
            nn.Linear(hidden_size, hidden_size),   
        )
                
        self.residual_nn = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.final_linear = nn.Linear(hidden_size * 3, vocab_size)
        
        
                
        # self.gru_nn = nn.GRU(hidden_size, hidden_size, 2, batch_first=False)
        # self.gru_linear = nn.Linear(hidden_size, vocab_size)
        # Around 50 % slower, but only 0.8% better
        
        # Dropouts makes it slightly worse
        
        # self.attention_nn = nn.Sequential(
        #     AttentionBlock(hidden_size, hidden_size),
        #     nn.Linear(hidden_size, vocab_size),
        # )
        
        # self.transformer_nn = nn.Sequential(
        #     TransformerBlock(hidden_size, num_heads=2, num_layers=2),  # Ensure num_heads is a factor of hidden_size
        #     nn.Linear(hidden_size, vocab_size),
        # ) SLOW
        
        # Ensemble -> little bit better but slower

        
        # We use this to reduce the precision of the input tensor for smaller model size
        self.quant = torch.quantization.QuantStub() if quantize else None 
        self.dequant = torch.quantization.DeQuantStub() if quantize else None

    def forward(self, x):
        if self.quant is not None:
            x = self.quant(x)
        embedded = self.embedding(x)
        
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        residual_out = self.residual_nn(embedded)
        
        combined_out = torch.cat((linear_out, dense_out, residual_out), dim=-1)
        
        final_out = self.final_linear(combined_out)
        
        if self.quant is not None:
            final_out = self.dequant(final_out)
        return final_out



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