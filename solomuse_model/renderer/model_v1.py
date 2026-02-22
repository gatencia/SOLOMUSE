import torch
import torch.nn as nn
import torch.nn.functional as F

class RendererConv1D_V1(nn.Module):
    """
    Baseline 1D Convolutional Neural Network for rendering encoded chunks.
    Maps encoded Situation [32] + Intent [7] + Backing Audio [C_x] -> Solo Audio [C_y] over F frames.
    """
    def __init__(self, c_x: int, d_int: int, d_sit: int, c_y: int, hidden_dim: int = 128, num_blocks: int = 3):
        super().__init__()
        
        self.c_x = c_x
        self.d_int = d_int
        self.d_sit = d_sit
        self.c_y = c_y
        
        in_channels = c_x + d_int + d_sit
        
        self.proj_in = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(0.1))
            
        self.blocks = nn.Sequential(*blocks)
        
        self.proj_out = nn.Conv1d(hidden_dim, c_y, kernel_size=1)
        
    def forward(self, x_encoded: torch.Tensor, intent: torch.Tensor, situation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_encoded: [B, F, C_x] Backing codes
            intent: [B, F, D_int] Sequence
            situation: [B, F, D_sit] Situational tracker (broadcasted)
            
        Returns:
            [B, F, C_y] target codes
        """
        # Conv1d expects [B, Channels, Length]
        x_c = x_encoded.transpose(1, 2)
        int_c = intent.transpose(1, 2)
        sit_c = situation.transpose(1, 2)
        
        # Concatenate along channel dim
        feat = torch.cat([x_c, int_c, sit_c], dim=1)
        
        h = self.proj_in(feat)
        h = self.blocks(h)
        out = self.proj_out(h)
        
        # Return [B, F, C_y]
        return out.transpose(1, 2)
