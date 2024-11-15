import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_size = self.hidden_size // self.num_heads
        self.dropout = config.dropout

        # Create linear layers for queries, keys, values
        self.q_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Linear transformations
        queries = self.q_linear(x)
        keys = self.k_linear(x)
        values = self.v_linear(x)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, values)

        # Reshape and apply output transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.out_linear(context)
        output = self.resid_dropout(output)

        return output
