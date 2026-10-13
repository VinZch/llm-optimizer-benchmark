"""Bounded-workspace persistent state compression around pinned optimizer formulas.

Each native parameter update sees FP32 working states. Only one parameter's selected
states are decoded at a time. Persistent state contains payload/scales, never an FP32
shadow. Group clocks are committed once per original group; model clocks once per step.
Prodigy's coupled reduction has a separate audited two-pass transformation below.
"""
import ast
import copy
from contextlib import contextmanager
import inspect
import textwrap
import types
import torch
from quantization.adamw import encode,decode
from quantization.state_simulation import StateSimulation,variants

SUPPORTED={'lion','signum','cadamw','adopt','ademamix','adamw-magma','sf-adamw','prodigy',
           'mars-adamw','mars-lion','mars-shampoo','sophiag','muon','d-muon','muon-magma',
           'softeq-k2000-muon','soap','newton-muon'}

class PackedOptimizer:
    # Role selection is shared; no simulation hooks are registered.
    role=StateSimulation.role
    def __init__(self,opt,name,variant):
        if name not in SUPPORTED: raise ValueError('Packed adapter not validated for '+name)
        self.opt,self.name,self.variant=opt,name,variant
        self.first,self.second=variants(name)[variant]
        self.active_parameters=0; self.max_active_parameters=0; self.max_decoded_numel=0
        self.native_step=opt.step
        self.native_state_dict=opt.state_dict
        self.native_load_state_dict=opt.load_state_dict
        self.native_train=getattr(opt,'train',None); self.native_eval=getattr(opt,'eval',None)
        self.native_hessian=getattr(opt,'update_hessian',None)
        self.clock_names=[k for k in ['global_step','step_num'] if hasattr(opt,k)]
        if name=='prodigy': self.prodigy_step=self._make_prodigy_step()
        controller=self
        def step(native_self,*args,**kwargs): return controller.step(*args,**kwargs)
        def state_dict(native_self): return controller.state_dict()
        def load_state_dict(native_self,state): return controller.load_state_dict(state)
        opt.step=types.MethodType(step,opt)
        opt.state_dict=types.MethodType(state_dict,opt)
        opt.load_state_dict=types.MethodType(load_state_dict,opt)
        if self.native_train:
            def train(native_self): return controller._dispatch(controller.native_train,'train')
            def eval(native_self): return controller._dispatch(controller.native_eval,'eval')
            opt.train=types.MethodType(train,opt); opt.eval=types.MethodType(eval,opt)
        if self.native_hessian:
            def update_hessian(native_self): return controller._dispatch(controller.native_hessian,'hessian')
            opt.update_hessian=types.MethodType(update_hessian,opt)

    def _written(self,key,event):
        if event in {'train','eval','init','readonly'}: return False
        if self.name=='sophiag': return (key=='hessian')==(event=='hessian')
        return True

    @contextmanager
    def parameter(self,p,event):
        if event in {'step','hessian'} and p.grad is None: event='readonly'
        state=self.opt.state[p]; saved={}; decoded=0
        self.active_parameters+=1
        self.max_active_parameters=max(self.max_active_parameters,self.active_parameters)
        if self.active_parameters!=1: raise RuntimeError('More than one decoded parameter')
        try:
            for key,value in list(state.items()):
                if self.role(p,key) is None: continue
                if not isinstance(value,dict) or 'payload' not in value: continue
                saved[key]=value
                needs_read=event!='init'
                if event in {'train','eval'}: needs_read=key=='z'
                if event=='hessian': needs_read=key=='hessian'
                if needs_read:
                    state[key]=decode(value); decoded+=p.numel()
            self.max_decoded_numel=max(self.max_decoded_numel,decoded)
            yield
        finally:
            for key,value in list(state.items()):
                role=self.role(p,key)
                if role is None: continue
                if key in saved and not self._written(key,event):
                    state[key]=saved[key]
                elif isinstance(value,torch.Tensor):
                    if value.dtype!=torch.float32: raise TypeError('Native persistent state is not FP32')
                    writes=saved[key]['writes']+1 if key in saved else 1
                    state[key]=encode(value,role[1],writes)
            self.active_parameters-=1

    @torch.no_grad()
    def _dispatch(self,method,event,*args,**kwargs):
        groups=self.opt.param_groups
        clocks={k:getattr(self.opt,k) for k in self.clock_names}
        after_clocks=clocks.copy(); loss=None
        # Newton refresh must see the complete map exactly once, in upstream order.
        original_flag=None
        if self.name=='newton-muon' and event=='step':
            original_flag=self.opt.precond_flag_for_step
            if original_flag(): self.opt._refresh_preconditioners()
            self.opt.precond_flag_for_step=lambda *a,**kw: False
        try:
            for group in groups:
                before={k:v for k,v in group.items() if k!='params'}
                after=before.copy()
                parameters=list(group['params'])
                if self.name in {'muon','d-muon','muon-magma','newton-muon','softeq-k2000-muon'}:
                    # Upstream visits every matrix branch before its AdamW backup;
                    # preserve this order for Magma's random mask stream.
                    parameters.sort(key=lambda p:not self.opt.state[p].get('use_muon',False))
                if not parameters:
                    self.opt.param_groups=[dict(before,params=[])]; method(*args,**kwargs)
                    after={k:v for k,v in self.opt.param_groups[0].items() if k!='params'}
                for p in parameters:
                    for key,value in clocks.items(): setattr(self.opt,key,value)
                    temporary=dict(before,params=[p])
                    self.opt.param_groups=[temporary]
                    # Read-only mode transforms include parameters whose grad is None.
                    with self.parameter(p,event): loss=method(*args,**kwargs)
                    after={k:v for k,v in temporary.items() if k!='params'}
                    after_clocks={k:getattr(self.opt,k) for k in self.clock_names}
                group.update(after)
        finally:
            self.opt.param_groups=groups
            if original_flag is not None: self.opt.precond_flag_for_step=original_flag
            for key,value in after_clocks.items(): setattr(self.opt,key,value)
        return loss

    @torch.no_grad()
    def step(self,*args,**kwargs):
        if args or kwargs.get('closure') is not None: raise ValueError('Closures are not supported by packed warmup')
        if self.name=='prodigy': return self.prodigy_step(self.opt)
        return self._dispatch(self.native_step,'step',*args,**kwargs)

    def _make_prodigy_step(self):
        """Move m/v recurrence to final pass: the d reduction does not read m/v.

        Use the OLD d for the recurrence and dlr, and the UPDATED d for epsilon,
        exactly as upstream. First-pass s/p0/reductions remain in their original order.
        This avoids either a full-model FP32 workspace or premature quantization.
        """
        function=inspect.unwrap(self.native_step.__func__)
        tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
        fn=tree.body[0]; fn.decorator_list=[]
        loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='p']
        if len(loops)!=2: raise RuntimeError('Pinned Prodigy loop layout changed')
        first,second=loops
        moved=[]
        class MoveMoments(ast.NodeTransformer):
            def visit_Expr(self,node):
                text=ast.unparse(node)
                if text.startswith(('exp_avg.mul_(', 'exp_avg_sq.mul_(')):
                    moved.append(copy.deepcopy(node)); return None
                return node
        first.body=[n for n in (MoveMoments().visit(n) for n in first.body) if n is not None]
        if len(moved)!=2: raise RuntimeError('Expected two independent moment recurrences')
        class OldD(ast.NodeTransformer):
            def visit_Name(self,node):
                return ast.copy_location(ast.Name(id='_packed_old_d',ctx=node.ctx),node) if node.id=='d' else node
        moved=[OldD().visit(n) for n in moved]
        for i,node in enumerate(second.body):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='denom' for t in node.targets):
                second.body[i:i]=[ast.If(test=ast.parse("group['lr'] > 0",mode='eval').body,body=moved,orelse=[])]; break
        else: raise RuntimeError('Missing final denominator')
        fn.body.insert(0,ast.parse("_packed_old_d = self.param_groups[0]['d']").body[0])
        for loop,event in [(first,'init'),(second,'step')]:
            context=ast.parse(f"_packed_controller.parameter(p, '{event}')",mode='eval').body
            loop.body=[ast.With(items=[ast.withitem(context_expr=context)],body=loop.body)]
        ast.fix_missing_locations(tree)
        namespace=dict(function.__globals__); namespace['_packed_controller']=self
        self.prodigy_transformed_source=ast.unparse(tree)
        exec(compile(tree,'<packed_prodigy>','exec'),namespace)
        return namespace['step']

    def state_dict(self):
        result=self.native_state_dict()
        result['packed_adapter']=dict(schema=1,optimizer=self.name,variant=self.variant,
            clocks={k:getattr(self.opt,k) for k in self.clock_names})
        return result

    def load_state_dict(self,state_dict):
        saved=copy.deepcopy(state_dict); metadata=saved.pop('packed_adapter')
        if (metadata['schema'],metadata['optimizer'],metadata['variant'])!=(1,self.name,self.variant): raise ValueError('Packed checkpoint identity mismatch')
        groups=saved['param_groups']; current=self.opt.param_groups
        if len(groups)!=len(current) or any(len(a['params'])!=len(b['params']) for a,b in zip(groups,current)): raise ValueError('Group mismatch')
        mapping={i:p for a,b in zip(groups,current) for i,p in zip(a['params'],b['params'])}
        def move(value,p):
            if isinstance(value,torch.Tensor): return value.to(device=p.device)
            if isinstance(value,dict): return {k:move(v,p) for k,v in value.items()}
            if isinstance(value,list): return [move(v,p) for v in value]
            if isinstance(value,tuple): return tuple(move(v,p) for v in value)
            return value
        self.opt.state.clear()
        for index,state in saved['state'].items(): self.opt.state[mapping[index]]=move(state,mapping[index])
        for a,b in zip(current,groups): a.update({k:v for k,v in b.items() if k!='params'})
        for key,value in metadata['clocks'].items(): setattr(self.opt,key,value)

    def inventory(self,named_parameters):
        rows=[]
        def visit(name,p,key,value):
            role=self.role(p,key)
            if isinstance(value,dict) and 'payload' in value:
                rows.append(dict(canonical_parameter_name=name,shape=list(value['shape']),branch=self.name,state_key=key,
                    role=role[0],actually_used=True,payload_bytes=value['payload'].numel()*value['payload'].element_size(),
                    metadata_bytes=0 if value['scales'] is None else value['scales'].numel()*value['scales'].element_size(),
                    stored_dtype=str(value['payload'].dtype),codec=value['codec'],writes=value['writes'],
                    protected_reason='numel<4096' if p.numel()<4096 else None))
            elif isinstance(value,torch.Tensor):
                rows.append(dict(canonical_parameter_name=name,shape=list(value.shape),branch=self.name,state_key=key,
                    role='protected',actually_used=not(self.name in {'mars-lion','mars-shampoo'} and p.ndim==2 and key=='exp_avg_sq'),
                    payload_bytes=value.numel()*value.element_size(),metadata_bytes=0,stored_dtype=str(value.dtype),codec='native',protected_reason='algorithm_specific'))
            elif isinstance(value,(tuple,list)):
                for i,item in enumerate(value): visit(name,p,f'{key}.{i}',item)
        for name,p in named_parameters:
            for key,value in self.opt.state[p].items(): visit(name,p,key,value)
        return rows
