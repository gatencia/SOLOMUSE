import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from solomuse_model.renderer.model_v2.transformer_lm import TokenTransformerRenderer
from solomuse_model.renderer.dataset_tokens import token_collate_fn
from unittest.mock import MagicMock

def test_model_forward_shapes():
    B, T, Q, Vocab = 2, 10, 4, 1024
    
    x_tokens = torch.randint(0, Vocab, (B, T, Q))
    y_tokens = torch.randint(0, Vocab, (B, T, Q))
    intent = torch.randn(B, T, 7)
    sit = torch.randn(B, 32)
    
    model = TokenTransformerRenderer(d_model=128, nhead=4, num_layers=2, 
                                     num_codebooks=Q, vocab_size=Vocab)
    
    logits = model(x_tokens, intent, sit, y_tokens)
    assert logits.shape == (B, T, Q, Vocab)

def test_token_dataset_padding():
    batch = [
        {
            "x_tokens": torch.zeros((5, 4), dtype=torch.long),
            "y_tokens": torch.ones((5, 4), dtype=torch.long),
            "intent_aligned": torch.randn(5, 7),
            "situation": torch.ones(32)
        },
        {
            "x_tokens": torch.zeros((10, 4), dtype=torch.long),
            "y_tokens": torch.ones((10, 4), dtype=torch.long),
            "intent_aligned": torch.randn(10, 7),
            "situation": torch.ones(32)
        }
    ]
    
    collated = token_collate_fn(batch)
    
    assert collated["x_tokens"].shape == (2, 10, 4)
    assert collated["lengths"][0] == 5

def test_inference_determinism():
    from solomuse_model.renderer.infer_v2 import TokenRendererSimulator
    
    # We mock init heavily here
    TokenRendererSimulator.__init__ = lambda self: None
    sim = TokenRendererSimulator()
    sim.cfg = MagicMock()
    sim.cfg.canonical_sample_rate = 24000
    sim.device = 'cpu'
    sim.num_codebooks = 4
    sim.vocab_size = 1024
    
    torch.manual_seed(42)
    sim.model = TokenTransformerRenderer(d_model=128, nhead=4, num_layers=2)
    sim.model.eval()
    
    class DummyAdapter:
        def decode(self, tokens, sr):
            return np.ones(5000, dtype=np.float32)
    sim.adapter = DummyAdapter()
    
    x_tok = np.zeros((10, 4), dtype=np.int32)
    i_al = np.zeros((10, 7), dtype=np.float32)
    sit = np.zeros(32, dtype=np.float32)
    
    y_tok, y_wav = sim.generate(x_tok, i_al, sit, temperature=0.0)
    
    assert y_tok.shape == (10, 4)
    assert y_wav.shape == (5000,)
