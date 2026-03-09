import torch
import torch.nn as nn

class MultiCodebookEmbedding(nn.Module):
    """
    Embeds [B, T, Q] discrete tokens into [B, T, D_model].
    Each of the Q codebooks gets its own embedding table, and their results are summed.
    """
    def __init__(self, num_codebooks: int, vocab_size: int, d_model: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        
        # We create a ModuleList of Q embedding layers
        self.embedders = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
            for _ in range(num_codebooks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, Q] integer indices
        Returns:
            out: [B, T, D_model] summed continuous embeddings
        """
        B, T, Q = x.shape
        if Q != self.num_codebooks:
            raise ValueError(f"Expected {self.num_codebooks} codebooks, got {Q}")
            
        out = 0
        for q in range(Q):
            # Extract [B, T] indices for codebook q and pass through its specific embedder
            out += self.embedders[q](x[:, :, q])
            
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D_model]
        Returns: [B, T, D_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]
