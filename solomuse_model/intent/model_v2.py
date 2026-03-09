import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard trigonometric positional encoding for injecting sequence order 
    information into the Transformer encoder.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class IntentPlannerTransformer_V2(nn.Module):
    """
    A streaming-friendly Autoregressive Transformer for sequence-to-sequence intent planning.
    
    Predicts intent targets (D_out=7) from situation context broadcasted over time (D_in=32).
    Uses a standard nn.TransformerEncoder with a strict causal mask to enforce temporal causality.
    """
    def __init__(self, 
                 input_dim: int = 32, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2, 
                 nhead: int = 8,
                 output_dim: int = 7, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. Input embedding: Map features [32] to model dim [128]
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 3. Transformer stack
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True # We will process [B, T, D] natively
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Output projection: Map hidden states back to target targets [7]
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize output linear weights explicitly to assist bounded activation
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Intent targets are bounded [0, 1] structurally
        self.activation = nn.Sigmoid()
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        This forces the Transformer self-attention to only look at past/current timesteps.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape [B, T, input_dim]
               B = batch size
               T = frame sequence length
               
        Returns:
            Tensor of shape [B, T, output_dim] representing soft intent targets bounded [0, 1]
        """
        B, T, _ = x.shape
        
        # Clamp massive outliers
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Project -> [B, T, hidden_dim]
        embedded = self.input_proj(x)
        
        # Positional Encoding expects [T, B, hidden_dim] normally, but we handle it manually
        # by reshaping to apply correctly since batch_first=True is set on the Encoder.
        embedded = embedded.transpose(0, 1) # [T, B, D]
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1) # Back to [B, T, D]
        
        # Generate causal mask for autoregressive visibility [T, T]
        causal_mask = self._generate_causal_mask(T, x.device)
        
        # Pass through Transformer Stack
        # is_causal parameter natively supported in later PyTorch versions, mask handles functionally
        # To avoid Apple Silicon MPS Causal Mask Segmentation Faults with -inf, we ensure fallback here
        memory = self.transformer_encoder(embedded, mask=causal_mask, is_causal=True)
        
        # memory: [B, T, hidden_dim]
        memory = torch.clamp(memory, min=-10.0, max=10.0)
        
        # Project logits: [B, T, output_dim]
        logits = self.output_proj(memory)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # Soft boundaries [0, 1]
        preds = self.activation(logits)
        
        return preds
