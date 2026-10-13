"""Serialize CPU snapshots so pickle cycles cannot retain old CUDA storage."""
import gc
import torch

def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value,dict):
        return {key:cpu_snapshot(item) for key,item in value.items()}
    if isinstance(value,list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value,tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value

def save_checkpoint_snapshot(value,path):
    snapshot=cpu_snapshot(value)
    torch.save(snapshot,path)
    # torch.save creates a local pickler class with cyclic storage references.
    gc.collect()
