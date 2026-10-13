"""GN outer update with the upstream objective and memory-bounded batches.

16,384 training targets per OUTER update; split evenly over upstream inner steps.
GN-full's original line search consumes an additional 16,384 targets per outer step.
These token semantics must be reported, not equated to one ordinary AdamW update.
"""
import torch
from optim.gn import (clone_param_dict,current_param_dict,compute_gn_step,
                      clone_param_dict_from_named_params,sub_param_dict,line_search_over_direction)

def gn_outer_step(model,opt,cfg,train,rng,batch,microbatch):
    if 32%cfg.gn_inner_iters: raise ValueError('GN inner count must divide 32 sequences')
    count=32//cfg.gn_inner_iters
    if count%microbatch: raise ValueError('GN microbatch must divide sequences per inner update')
    anchor=clone_param_dict(model); params=current_param_dict(model)
    loss=0.; norm=None
    for inner in range(cfg.gn_inner_iters):
        opt.zero_grad(set_to_none=True)
        starts=rng.integers(0,len(train)-513,size=count)
        for offset in range(0,count,microbatch):
            x,y=batch(train,starts[offset:offset+microbatch])
            with torch.autocast('cuda' if x.is_cuda else 'cpu',dtype=torch.bfloat16,enabled=x.is_cuda):
                grads,metrics=compute_gn_step(model,anchor,x,y,mode='full' if cfg.opt=='gn-full' else 'prox',prox_weight_decay=cfg.gn_inner_wd,moe=False)
            for p,g in zip(params.values(),grads):
                if p.grad is None: p.grad=g.detach().clone().div_(count//microbatch)
                else: p.grad.add_(g.detach(),alpha=1/(count//microbatch))
            loss+=metrics.base_loss/(cfg.gn_inner_iters*(count//microbatch))
            del grads
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        opt.step(); opt.zero_grad(set_to_none=True)
    extra=0; step_size=1.
    if cfg.gn_linesearch:
        current=clone_param_dict_from_named_params(params); direction=sub_param_dict(current,anchor)
        starts=rng.integers(0,len(train)-513,size=32)
        batches=[batch(train,starts[i:i+microbatch]) for i in range(0,32,microbatch)]
        with torch.autocast('cuda' if next(model.parameters()).is_cuda else 'cpu',dtype=torch.bfloat16,enabled=next(model.parameters()).is_cuda):
            step_size,_=line_search_over_direction(model,anchor,direction,batches,moe=False,ls_range=cfg.gn_ls_range)
        extra=16384
    return loss,norm,extra,step_size
