import torch
import torch.nn as nn
import torch.nn.functional as F


# IDEA: https://arxiv.org/pdf/1911.03572 (see Supporter model section)
class SupporterModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, quantize: bool = False, use_rnn: bool = False):
        super(SupporterModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear_nn = nn.Linear(hidden_size, hidden_size)
        self.dense_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),  # Appears to perform better than ReLU
            nn.Linear(hidden_size, hidden_size),   
        )
        self.use_rnn = use_rnn
        
        
        if use_rnn:    
            self.rnn_nn = nn.Sequential(
                RNNBlock(hidden_size, hidden_size, num_layers=1),
            )
        else:
            self.residual_nn = nn.Sequential(
                ResidualLinearBlock(hidden_size, hidden_size),
                ResidualLinearBlock(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
            )
        
        
        self.final_linear = nn.Linear(hidden_size * 3, vocab_size)
        
        self.lstm_block = LSTMBlock(hidden_size, hidden_size, num_layers=1)

        #self.rnn_nn = nn.RNN(hidden_size, hidden_size, num_layers=2, batch_first=True)
        
        
                
        # self.gru_nn = nn.GRU(hidden_size, hidden_size, 2, batch_first=False)
        # self.gru_linear = nn.Linear(hidden_size, vocab_size)
        # Around 50 % slower, but only 0.8% better
        
        # RNN -> very slow ( 53 times slower) and with very small (< 0.1%) improvement
        
        # Dropouts makes it slightly worse
        
        # self.attention_nn = nn.Sequential(
        #     AttentionBlock(hidden_size, hidden_size),
        #     nn.Linear(hidden_size, vocab_size),
        # )
        
        # self.transformer_nn = nn.Sequential(
        #     TransformerBlock(hidden_size, num_heads=2, num_layers=2),  # Ensure num_heads is a factor of hidden_size
        #     nn.Linear(hidden_size, hidden_size),
        # ) # SLOW
        
        # We use this to reduce the precision of the input tensor for smaller model size
        self.quant = torch.quantization.QuantStub() if quantize else None 
        self.dequant = torch.quantization.DeQuantStub() if quantize else None

    def forward(self, x):
        if self.quant is not None:
            x = self.quant(x)
        embedded = self.embedding(x)
        
        linear_out = self.linear_nn(embedded)
        dense_out = self.dense_nn(embedded)
        
        out = self.rnn_nn(embedded) if self.use_rnn else self.residual_nn(embedded)
    
        combined_out = torch.cat((linear_out, dense_out, out), dim=-1)
        
        final_out = self.final_linear(combined_out)
        
        
        if self.quant is not None:
            final_out = self.dequant(final_out)
        return final_out
    
    

# 4335
class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNBlock, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=False)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class LSTMBlock(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMBlock, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        def forward(self, x):
            out, _ = self.lstm(x)
            return out
    
class ResidualLinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ResidualLinearBlock, self).__init__()
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
 
class ConvBlock2(nn.Module):
    def __init__(self, ni):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv2d(ni, ni, 1)

    def forward(self, x):
        # Assuming x has shape (batch_size, seq_length, hidden_size)
        # Reshape x to (batch_size, hidden_size, seq_length, 1) to match Conv2d input requirements
        x = x.permute(0, 2, 1).unsqueeze(-1)
        out = F.relu(self.conv1(x))
        # Reshape back to (batch_size, seq_length, hidden_size)
        out = out.squeeze(-1).permute(0, 2, 1)
        return out

        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.activation = activation_fn

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)



    
    
class TransformerBlock(nn.Module):
    def __init__(self, input_size, num_heads=2, num_layers=2):
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
        with torch.no_grad():
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