"""Use the pinned upstream parser/factory without importing its training services."""
import ast
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'upstream/llm-optimizer-benchmark'
sys.path.insert(0, str(SOURCE / 'src'))

def factory_namespace():
    path = SOURCE / 'src/main.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'build_optimizer', 'build_adamw_optimizer', 'get_optimizer_param_list',
             'get_mup_width_mult', 'get_muon_effective_backup_lr', 'uses_combined_scheduler', 'build_scheduler'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import) and all(a.name in {'torch','inspect'} for a in node.names):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and (node.module or '').startswith('optim.') and node.module != 'optim.base':
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'MUON_SCHEDULER_OPTS','MAGMA_OPT_LABELS'} for t in node.targets):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

def config_for(name):
    from config.base import parse_args
    script = {'sophiag':'sophia','mars-adamw':'mars','mars-lion':'mars','mars-shampoo':'mars'}.get(name,name)
    text = (SOURCE / f'scripts/124m/{script}.sh').read_text()
    import re
    # Select the first launch recipe, ignoring logging, DDP and dataset arguments.
    pairs = re.findall(r'--([a-zA-Z0-9_]+)\s+((?!--)[^\s\\]+)', text)
    allowed = {'lr','weight_decay','beta1','beta2','momentum','nesterov','scheduler',
               'precondition_frequency','max_precond_dim','shampoo_beta','merge_dims',
               'precondition_1d','normalize_grads','correct_bias','sophia_rho','sophia_bs',
               'cautious_xi','magma_scope','magma_survival_p','magma_tau','magma_beta'}
    argv = []
    seen = set()
    for key,value in pairs:
        if key in seen: continue
        if key in allowed or key.startswith(('mars_','adema_','adopt_','prodigy_','muon_','newton_muon_','gn_')):
            argv += ['--'+key, value]; seen.add(key)
    opt = 'mars' if name.startswith('mars-') else name
    argv += ['--model','llama','--opt',opt, '--device','cuda',
             '--n_embd','768','--n_head','12','--n_layer','12','--sequence_length','512',
             '--iterations','91553','--warmup_steps','3000','--batch_size','2','--acc_steps','16']
    cfg = parse_args(argparse.ArgumentParser(), argv, None)
    cfg.distributed_backend=None
    cfg.gn_linesearch='--gn_linesearch' in text
    if name.startswith('mars-'): cfg.mars_type=name
    cfg.world_size=1
    return cfg
