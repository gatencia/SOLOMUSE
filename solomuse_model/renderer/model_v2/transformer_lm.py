import torch
import torch.nn as nn
from typing import Optional

from .embeddings import MultiCodebookEmbedding, PositionalEncoding

class TokenTransformerRenderer(nn.Module):
    """
    Autoregressive Transformer LM for rendering multiple discrete codebook tokens.
    Conditioned on backing tokens (x), continuous intent, and situation vector.
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, 
                 ffn_dim: int = 2048,
                 num_codebooks: int = 4, vocab_size: int = 1024, 
                 d_intent: int = 7, d_situation: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
 
        # --- Embedders ---
        self.x_embedder = MultiCodebookEmbedding(num_codebooks, vocab_size, d_model)
        self.y_embedder = MultiCodebookEmbedding(num_codebooks, vocab_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Project continuous conditions into D_model
        self.intent_proj = nn.Linear(d_intent, d_model)
        self.sit_proj = nn.Linear(d_situation, d_model)
        
        # --- Transformer Core ---
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=ffn_dim, dropout=dropout, 
                                                   batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # --- LM Head ---
        # Output is [B, T, Q * Vocab_Size] then reshaped
        self.out_head = nn.Linear(d_model, num_codebooks * vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, 
                x_tokens: torch.Tensor, 
                intent_aligned: torch.Tensor, 
                situation: torch.Tensor, 
                y_tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculates logits for the entire sequence (teacher forcing).
        
        Args:
           x_tokens: [B, T, Q] backing context
           intent_aligned: [B, T, D_int] upsampled sequencing
           situation: [B, D_sit] structural style mapping
           y_tokens: [B, T, Q] target track
           
        Returns:
           logits: [B, T, Q, Vocab_Size]
        """
        B, T, Q = y_tokens.size()
        
        # 1. Embed discrete inputs
        x_emb = self.x_embedder(x_tokens) # [B, T, d_model]
        y_emb = self.y_embedder(y_tokens) # [B, T, d_model]
        
        # 2. Embed continuous inputs
        int_emb = self.intent_proj(intent_aligned) # [B, T, d_model]
        
        # broadcast situation across time dimension
        sit_emb = self.sit_proj(situation).unsqueeze(1).expand(B, T, self.d_model) # [B, T, d_model]
        
        # 3. Create context mapping (Memory for decoder)
        # Summing embeddings acts as a unified prompt. Alternatively, concatenating creates very long sequences.
        memory = x_emb + int_emb + sit_emb 
        memory = self.pos_encoder(memory)
        
        # 4. Target Sequence Masking
        # Causal mask enforces y_t only attends to y_{t-1} and before
        tgt_mask = self.generate_causal_mask(T, device=y_tokens.device)
        y_embedded = self.pos_encoder(y_emb)
        
        # 5. Transformer Pass
        # memory is cross-attention source. tgt is self-attention sequence
        hidden = self.transformer(tgt=y_embedded, memory=memory, 
                                  tgt_mask=tgt_mask)
                                  
        # 6. Output mapping
        # [B, T, Q * V]
        logits_flat = self.out_head(hidden)
        
        # reshape to [B, T, Q, VocabSize]
        logits = logits_flat.view(B, T, self.num_codebooks, self.vocab_size)
        
        return logits
