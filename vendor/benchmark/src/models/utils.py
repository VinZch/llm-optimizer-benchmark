"""Minimal registry for the unmodified upstream dense Llama smoke runner.

Adapted from benchmark commit 7776f6b0b189707911d43f11853c40851fd058a8.
Other model imports omitted because this runner only supports dense Llama.
"""
import torch
from .base import LayerNorm
from .llama import RMSNorm
BLACKLIST_WEIGHT_MODULES = (torch.nn.LayerNorm, LayerNorm, RMSNorm, torch.nn.Embedding)
