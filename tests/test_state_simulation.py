import copy
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from benchmark_bridge import factory_namespace,config_for
from quantization.state_simulation import StateSimulation,variants
from quantization.adamw import encode,decode

class SimulationTests(unittest.TestCase):
    def make(self,name):
        cfg=config_for(name); cfg.device='cpu'; cfg.max_precond_dim=32
        model=torch.nn.Linear(64,64,bias=False)
        opt=factory_namespace()['build_optimizer'](cfg,model,[dict(params=list(model.parameters()))],{id(p) for p in model.parameters()})
        if name=='sf-adamw': opt.train()
        return model,opt

    def test_identity_matches_native_and_roundtrip(self):
        for name in ['lion','signum','cadamw','adopt','ademamix','adamw-magma','sf-adamw','prodigy','mars-adamw','mars-lion','mars-shampoo','sophiag','muon','d-muon','muon-magma','softeq-k2000-muon','soap']:
            with self.subTest(optimizer=name):
                torch.manual_seed(2); m0,o0=self.make(name)
                torch.manual_seed(2); m1,o1=self.make(name)
                sim=StateSimulation(o1,name,next(iter(variants(name))))
                for step in range(4):
                    g=torch.randn_like(m0.weight)*.02
                    for m,o in [(m0,o0),(m1,o1)]:
                        torch.manual_seed(100+step); m.weight.grad=g.clone(); o.step()
                        if name=='sophiag': o.update_hessian()
                    self.assertTrue(torch.equal(m0.weight,m1.weight),name)
                if name=='sf-adamw':
                    before=copy.deepcopy(sim.writes); o0.eval(); o1.eval(); o0.train(); o1.train()
                    self.assertEqual(before,sim.writes)
                    self.assertTrue(torch.equal(m0.weight,m1.weight))

    def test_rounding_propagates_and_protects(self):
        for name in ['lion','cadamw','ademamix','prodigy','mars-adamw','sophiag']:
            torch.manual_seed(4); model,opt=self.make(name)
            variant='L2' if name=='lion' else ('H6' if name=='sophiag' else 'A6')
            sim=StateSimulation(opt,name,variant)
            for step in range(3):
                model.weight.grad=torch.randn_like(model.weight)*.03; opt.step()
                if name=='sophiag':
                    before=sim.writes.get((id(model.weight),'hessian'),0)
                    opt.update_hessian(); sim.round_states('hessian')
                    self.assertEqual(sim.writes[(id(model.weight),'hessian')],before+1)
                self.assertTrue(torch.isfinite(model.weight).all())
            state=opt.state[model.weight]
            key='exp_avg_fast' if name=='ademamix' else 'exp_avg'
            value=state[key]
            torch.testing.assert_close(value,decode(encode(value,'fp8')),rtol=1e-6,atol=1e-8)
            for protected in ['exp_avg_slow','s','p0','last_grad']:
                if protected in state: self.assertIsNone(sim.role(model.weight,protected))

if __name__=='__main__': unittest.main()
