import sys
import os
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
os.environ['TIKTOKEN_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'.tmp')
from benchmark_bridge import config_for
from models.llama import Llama
from optim.gn import clone_param_dict,compute_gn_step

class GNAccumulationTest(unittest.TestCase):
    def test_microbatch_gradients_equal_full_batch(self):
        torch.set_num_threads(2)
        for name in ['gn-prox','gn-full']:
            with self.subTest(mode=name):
                cfg=config_for(name); cfg.n_embd=16; cfg.n_head=2; cfg.n_layer=1
                cfg.multiple_of=16; cfg.vocab_size=32; cfg.sequence_length=4; cfg.dropout=0.; cfg.device='cpu'
                torch.manual_seed(5); model=Llama(cfg)
                anchor=clone_param_dict(model)
                with torch.no_grad():
                    for p in model.parameters(): p.add_(torch.randn_like(p)*1e-4)
                x=torch.randint(0,32,(2,4)); y=torch.randint(0,32,(2,4))
                mode=name.split('-')[1]
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    full,_=compute_gn_step(model,anchor,x,y,mode,0.)
                    halves=[compute_gn_step(model,anchor,x[i:i+1],y[i:i+1],mode,0.)[0] for i in range(2)]
                for a,b,c in zip(full,*halves): torch.testing.assert_close(a,(b+c)/2,rtol=3e-4,atol=3e-6)

if __name__=='__main__': unittest.main()
