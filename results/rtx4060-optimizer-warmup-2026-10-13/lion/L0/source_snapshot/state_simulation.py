"""Explicit state-rounding simulation; no packed-storage or speedup claims."""
import torch
from quantization.adamw import encode, decode

A = [('fp32','fp32'),('bf16','fp32'),('fp32','bf16'),('fp8','fp32'),('fp32','fp8'),('fp8','bf16'),('fp8','fp8')]

def variants(name):
    if name in {'lion','signum','muon','d-muon','muon-magma','newton-muon','softeq-k2000-muon','mars-lion','mars-shampoo'}:
        return {f'L{i}': (c,'fp32') for i,c in enumerate(('fp32','bf16','fp8'))}
    if name=='sf-adamw':
        return {f'S{i}':c for i,c in enumerate([('fp32','fp32'),('fp32','bf16'),('fp32','fp8'),('bf16','fp32'),('bf16','bf16')])}
    return {f'{"H" if name=="sophiag" else "A"}{i}':c for i,c in enumerate(A)}

class StateSimulation:
    def __init__(self, opt, name, variant):
        self.opt,self.name,self.variant=opt,name,variant
        self.first,self.second=variants(name)[variant]
        self.writes={}
        self.handle=opt.register_step_post_hook(lambda *_: self.round_states('step'))

    def role(self,p,key):
        if self.name.startswith('mars-'):
            if p.ndim!=2: return None
            if key=='exp_avg': return ('m',self.first)
            if key=='exp_avg_sq' and self.name=='mars-adamw': return ('v',self.second)
            return None
        if self.name in {'muon','d-muon','muon-magma','newton-muon','softeq-k2000-muon'}:
            if self.opt.state[p].get('use_muon') and key in {'muon_buffer','momentum_buffer'}:
                return ('matrix_momentum',self.first)
            return None
        keys = {'lion':('exp_avg',None),'signum':('momentum_buffer',None),
                'ademamix':('exp_avg_fast','exp_avg_sq'),
                'sf-adamw':('z','exp_avg_sq'),'sophiag':('exp_avg','hessian')}
        first,second=keys.get(self.name,('exp_avg','exp_avg_sq'))
        if key==first: return ('z' if key=='z' else 'm',self.first)
        if key==second: return ('h' if key=='hessian' else 'v',self.second)
        return None

    @torch.no_grad()
    def round_states(self,event):
        for group in self.opt.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                for key,value in self.opt.state[p].items():
                    role=self.role(p,key)
                    if role is None or not isinstance(value,torch.Tensor): continue
                    if self.name=='sophiag' and ((key=='hessian') != (event=='hessian')): continue
                    # Identity leaves state objects and lifecycle untouched.
                    token=(id(p),key); self.writes[token]=self.writes.get(token,0)+1
                    if role[1]!='fp32' and value.numel()>=4096:
                        value.copy_(decode(encode(value.float(),role[1])))

    def inventory(self,named_parameters):
        result=[]
        def visit(name,p,key,value):
            if isinstance(value,torch.Tensor):
                role=self.role(p,key)
                result.append(dict(canonical_parameter_name=name,shape=list(value.shape),branch=self.name,
                    state_key=key,role=role[0] if role else 'protected',actually_used=True,
                    payload_bytes=value.numel()*value.element_size(),metadata_bytes=0,
                    stored_dtype=str(value.dtype),codec='simulation_'+(role[1] if role else 'fp32'),
                    protected_reason='numel<4096' if value.numel()<4096 else ('algorithm_specific' if not role else None),
                    writes=self.writes.get((id(p),key),0)))
            elif isinstance(value,(list,tuple)):
                for i,v in enumerate(value): visit(name,p,f'{key}.{i}',v)
        for name,p in named_parameters:
            for key,value in self.opt.state[p].items(): visit(name,p,key,value)
        return result
