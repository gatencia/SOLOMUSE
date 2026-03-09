from solomuse_data.config import load_config
import solomuse_model.renderer.train_v2 as t_v2
import torch
import os

cfg = load_config("config.yaml")
cfg.renderer_batch_size = 2
cfg.renderer_epochs = 1
cfg.renderer_overfit_one_batch = True

# Overwrite device resolution function to force CPU exclusively
old_device = torch.device
def force_cpu(*args, **kwargs):
    return old_device('cpu')
torch.device = force_cpu

t_v2.run_train_renderer_v2(cfg, "slakh")
