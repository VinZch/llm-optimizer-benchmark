import copy
import io
import os
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'.tmp')
from benchmark_bridge import config_for,factory_namespace
from quantization.packed_optimizer import PackedOptimizer
from quantization.state_simulation import StateSimulation,variants
from quantization.adamw import decode
from models.llama import Llama

NAMES=['lion','signum','cadamw','adopt','ademamix','adamw-magma','sf-adamw','prodigy',
       'mars-adamw','mars-lion','mars-shampoo','sophiag','muon','d-muon','muon-magma','softeq-k2000-muon','soap','newton-muon']

def make(name,kind,variant,device='cpu'):
    device=os.environ.get('PACKED_TEST_DEVICE',device)
    cfg=config_for(name); cfg.device=device; cfg.max_precond_dim=64; cfg.precondition_frequency=2
    if name=='newton-muon':
        cfg.n_embd=16; cfg.n_head=2; cfg.n_layer=1; cfg.multiple_of=16; cfg.vocab_size=256; cfg.sequence_length=4; cfg.dropout=0.
        cfg.newton_muon_precond_every=2
        model=Llama(cfg).to(device)
    else:
        model=torch.nn.Sequential(torch.nn.Linear(64,64),torch.nn.Linear(64,64)).to(device)
    params=list(model.parameters()); groups=[dict(params=params[:len(params)//2]),dict(params=params[len(params)//2:])]
    # Muon creates its own groups, other optimizers retain the two group clocks.
    opt=factory_namespace()['build_optimizer'](cfg,model,groups,{id(p) for p in params})
    adapter=PackedOptimizer(opt,name,variant) if kind=='packed' else (StateSimulation(opt,name,variant) if kind=='simulation' else None)
    if name=='sf-adamw': opt.train()
    return model,opt,adapter

class PackedTests(unittest.TestCase):
    def test_256_update_replay(self):
        torch.set_num_threads(2)
        for name,variant in [('lion','L2'),('prodigy','A5'),('sf-adamw','S4'),('sophiag','H6')]:
            with self.subTest(optimizer=name):
                torch.manual_seed(44); m0,o0,sim=make(name,'simulation',variant)
                torch.manual_seed(44); m1,o1,pack=make(name,'packed',variant)
                for step in range(256):
                    grads=[torch.randn_like(p)*.01 for p in m0.parameters()]
                    for model,opt,adapter in [(m0,o0,sim),(m1,o1,pack)]:
                        for p,g in zip(model.parameters(),grads): p.grad=g.clone()
                        opt.step()
                        if name=='sophiag' and step%10==9:
                            opt.update_hessian()
                            if adapter is sim: sim.round_states('hessian')
                    for a,b in zip(m0.parameters(),m1.parameters()): torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)

    def test_native_identity_all_optimizers(self):
        torch.set_num_threads(2)
        for name in NAMES:
            with self.subTest(optimizer=name):
                variant=next(iter(variants(name)))
                torch.manual_seed(3); m0,o0,_=make(name,'native',variant)
                torch.manual_seed(3); m1,o1,adapter=make(name,'packed',variant)
                for step in range(4):
                    grads=[torch.randn_like(p)*.02 for p in m0.parameters()]
                    for m,o in [(m0,o0),(m1,o1)]:
                        torch.manual_seed(500+step)
                        for p,g in zip(m.parameters(),grads): p.grad=g.clone()
                        o.step()
                        if name=='sophiag': o.update_hessian()
                    for a,b in zip(m0.parameters(),m1.parameters()): torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
                self.assertEqual(adapter.max_active_parameters,1)
                for a,b in zip(o0.param_groups,o1.param_groups):
                    for key in ['k','step','d','d_max','d_numerator','weight_sum']:
                        if key in a: self.assertEqual(a[key],b[key],(name,key))

    def test_packed_matches_recurrent_simulation_all_variants(self):
        torch.set_num_threads(2)
        for name in NAMES:
            for variant in variants(name):
                with self.subTest(optimizer=name,variant=variant):
                    torch.manual_seed(7); m0,o0,sim=make(name,'simulation',variant)
                    torch.manual_seed(7); m1,o1,pack=make(name,'packed',variant)
                    for step in range(4):
                        grads=[torch.randn_like(p)*.01 for p in m0.parameters()]
                        for m,o,adapter in [(m0,o0,sim),(m1,o1,pack)]:
                            torch.manual_seed(200+step)
                            for p,g in zip(m.parameters(),grads): p.grad=g.clone()
                            o.step()
                            if name=='sophiag':
                                o.update_hessian()
                                if adapter is sim: sim.round_states('hessian')
                        for a,b in zip(m0.parameters(),m1.parameters()): torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
                        for p0,p1 in zip(m0.parameters(),m1.parameters()):
                            for key,value in o1.state[p1].items():
                                if isinstance(value,dict) and 'payload' in value:
                                    torch.testing.assert_close(o0.state[p0][key],decode(value),rtol=2e-6,atol=2e-7)
                    if name=='sf-adamw':
                        counts={(id(p),k):s['writes'] for p,st in o1.state.items() for k,s in st.items() if isinstance(s,dict) and 'writes' in s}
                        o0.eval(); o1.eval(); o0.train(); o1.train()
                        for a,b in zip(m0.parameters(),m1.parameters()): torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
                        self.assertEqual(counts,{(id(p),k):s['writes'] for p,st in o1.state.items() for k,s in st.items() if isinstance(s,dict) and 'writes' in s})

    def test_storage_checkpoint_and_resume(self):
        torch.set_num_threads(2)
        for name in ['lion','adopt','sf-adamw','prodigy','sophiag','newton-muon']:
            variant={'lion':'L2','sf-adamw':'S4','sophiag':'H6','newton-muon':'L2'}.get(name,'A6')
            torch.manual_seed(13); m0,o0,pack=make(name,'packed',variant)
            for p in m0.parameters(): p.grad=torch.randn_like(p)*.01
            o0.step()
            serialized=io.BytesIO(); torch.save(o0.state_dict(),serialized); serialized.seek(0)
            saved=torch.load(serialized,map_location='cpu',weights_only=False)
            torch.manual_seed(13); m1,o1,_=make(name,'packed',variant)
            m1.load_state_dict(m0.state_dict()); o1.load_state_dict(saved)
            for p0,p1 in zip(m0.parameters(),m1.parameters()): p0.grad=torch.ones_like(p0)*.001; p1.grad=p0.grad.clone()
            o0.step(); o1.step()
            for p0,p1 in zip(m0.parameters(),m1.parameters()): torch.testing.assert_close(p0,p1,rtol=0,atol=0)
            low=[s for state in o1.state.values() for s in state.values() if isinstance(s,dict) and s.get('codec') in ['fp8','bf16']]
            self.assertTrue(low)
            self.assertTrue(all(s['payload'].element_size()<4 for s in low))
            self.assertEqual(pack.max_active_parameters,1)
            self.assertLessEqual(pack.max_decoded_numel,2*max(p.numel() for p in m0.parameters()))

if __name__=='__main__': unittest.main()
