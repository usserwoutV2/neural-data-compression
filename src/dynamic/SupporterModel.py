import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class LSTMCellWithNorm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellWithNorm, self).__init__()
        self.hidden_size = hidden_size
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wj = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.layernorm_f = LayerNorm(hidden_size)
        self.layernorm_i = LayerNorm(hidden_size)
        self.layernorm_o = LayerNorm(hidden_size)
        self.layernorm_j = LayerNorm(hidden_size)
        self.layernorm_c = LayerNorm(hidden_size)

    def forward(self, x, prev_h, prev_c):
        combined = torch.cat((x, prev_h), dim=-1)

        ft = torch.sigmoid(self.layernorm_f(self.Wf(combined)))
        it = torch.sigmoid(self.layernorm_i(self.Wi(combined)))
        ot = torch.sigmoid(self.layernorm_o(self.Wo(combined)))
        jt = torch.tanh(self.layernorm_j(self.Wj(combined)))

        c = ft * prev_c + torch.min(1 - ft, it) * jt
        c = self.layernorm_c(c)
        h = ot * c

        return h, c
    
# IDEA: https://arxiv.org/pdf/1911.03572 (see Supporter model section)
class SupporterModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, quantize: bool = False):
        super(SupporterModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear_nn = nn.Linear(hidden_size, vocab_size)
        self.dense_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(), # Appears to perform better than ReLU
            nn.Linear(hidden_size, vocab_size),   
        )
                
        self.residual_nn = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )
        
        num_layers = 3
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList([
            LSTMCellWithNorm(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Final output layer to produce probabilities
        self.output_layer = nn.Linear(num_layers * hidden_size, vocab_size)
        
        # self.gru_nn = nn.GRU(hidden_size, hidden_size, 2, batch_first=False)
        # self.gru_linear = nn.Linear(hidden_size, vocab_size)
        # Around 50 % slower, but only 0.8% better
        
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
        

        
        x = linear_out + dense_out + residual_out 
        if self.quant is not None:
            x = self.dequant(x)
        return x



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