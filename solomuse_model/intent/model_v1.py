import torch
import torch.nn as nn

class IntentPlannerGRU_V1(nn.Module):
    """
    A baseline bidirectional GRU for sequence-to-sequence intent planning.
    
    Predicts intent targets (D_out=7) from situation context broadcasted over time (D_in=32).
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, num_layers: int = 2, output_dim: int = 7, dropout: float = 0.1):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        # Intent targets are bounded [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape [B, F, input_dim]
               B = batch size
               F = frame sequence length
               
        Returns:
            Tensor of shape [B, F, output_dim]
        """
        # gru_out: [B, F, hidden_dim * 2]
        gru_out, _ = self.gru(x)
        
        # logits: [B, F, output_dim]
        logits = self.linear(gru_out)
        
        # predictions: [B, F, output_dim] in [0, 1]
        preds = self.activation(logits)
        
        return preds
