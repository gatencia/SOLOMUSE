import torch
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from solomuse_model.renderer.model_v2.transformer_lm import TokenTransformerRenderer

print("Testing MPS operations...")
try:
    device = torch.device('mps')
    print(f"Device: {device}")
    
    B, T, Q, Vocab = 2, 450, 4, 1024
    
    x_tokens = torch.randint(0, Vocab, (B, T, Q), device=device)
    y_tokens = torch.randint(0, Vocab, (B, T, Q), device=device)
    intent = torch.randn(B, T, 7, device=device)
    sit = torch.randn(B, 32, device=device)
    
    model = TokenTransformerRenderer(d_model=128, nhead=4, num_layers=2, 
                                     num_codebooks=Q, vocab_size=Vocab).to(device)
    print("Model initialized on MPS")
    
    logits = model(x_tokens, intent, sit, y_tokens)
    print("Forward pass successful")
    
    logits_flat = logits.view(-1, Vocab)
    y_tokens_flat = y_tokens.view(-1)
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits_flat, y_tokens_flat)
    print("Loss pass successful")
    
    loss.backward()
    print("Backward pass successful")
    
except Exception as e:
    print(f"Exception: {e}")
