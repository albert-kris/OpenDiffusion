import torch
import torch.nn as nn

def init_xavier(m):
    """
    Initialize module weights using Xavier/Glorot uniform initialization.
    Applies to Conv2d and Linear layers, setting biases to zero if present.
    
    :param m: PyTorch module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_kaiming(m):
    """
    Initialize module weights using Kaiming/He uniform initialization.
    Applies to Conv2d and Linear layers, setting biases to zero if present.
    Recommended for networks using ReLU activations.
    
    :param m: PyTorch module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_orthogonal(m):
    """
    Initialize module weights using orthogonal initialization.
    Applies to Conv2d and Linear layers; biases are set to zero if present.

    :param m: PyTorch module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_normal(m):
    """
    Initialize module weights from a normal distribution (mean=0, std=0.02).
    Applies to Conv2d and Linear layers; biases are set to zero if present.

    :param m: PyTorch module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_trunc_normal(m):
    """
    Initialize module weights using a truncated normal distribution.
    Tries to use ``nn.init.trunc_normal_`` if available, otherwise falls
    back to a clipped normal. Biases are set to zero if present.

    :param m: PyTorch module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        try:
            # PyTorch provides trunc_normal_ in newer versions
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        except Exception:
            # Fallback: normal + clamp to approximate truncation
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                m.weight.clamp_(-0.04, 0.04)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_constant(m):
    """
    Initialize module weights to a small constant (0.01) and biases to zero.
    Useful for quick control experiments or when a small non-zero starting
    weight is desired.

    :param m: PyTorch module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.constant_(m.weight, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)