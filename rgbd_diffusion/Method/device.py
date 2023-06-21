import torch


def move_to(d, device):
    return {
        k: v.to(device=device) if isinstance(v, torch.Tensor) else v
        for k, v in d.items()
    }
