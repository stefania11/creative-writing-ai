import torch
import torch.nn as nn
import torch.nn.functional as F
from templates.nanoGPT_lite.attention import MultiHeadAttention

class Config:
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class CreativeWritingTransformer(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=768, n_head=12, n_layer=12, dropout=0.1):
        super().__init__()
        self.config = Config(vocab_size, n_embd, n_head, n_layer, dropout)

        # Initialize tokenizer first
        self.tokenizer = self.get_tokenizer()

        # Token embeddings with proper padding token
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=self.tokenizer.pad_token_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, n_embd))
        self.drop = nn.Dropout(dropout)

        # Layer normalization before transformer blocks
        self.ln_0 = nn.LayerNorm(n_embd)

        # Transformer blocks with gradient checkpointing for memory efficiency
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(n_layer)])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights with smaller standard deviation
        self.apply(self._init_weights)

        # Scale final layer weights
        with torch.no_grad():
            self.head.weight.data.normal_(mean=0.0, std=0.01)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Get token embeddings and apply initial normalization
        tok_emb = self.tok_emb(idx)
        tok_emb = self.ln_0(tok_emb)

        # Add positional embeddings
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop(tok_emb + pos_emb)

        # Apply transformer blocks with residual connections
        for block in self.blocks:
            x = x + block(x)

        # Final normalization and projection
        x = self.ln_f(x)
        logits = self.head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )

        return logits, loss if targets is not None else logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]

            attention_mask = (idx_cond != self.tokenizer.pad_token_id).float()

            with torch.no_grad():
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                logits[:, self.tokenizer.pad_token_id] = float('-inf')

                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                if idx_next.item() == self.tokenizer.pad_token_id:
                    break

            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next.item() == self.tokenizer.eos_token_id:
                break

        return idx

    def get_tokenizer(self):
        """Return the model's tokenizer"""
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained('gpt2')
