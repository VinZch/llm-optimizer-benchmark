"""Publish only this experiment's known sources and outputs to the authorized repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'target-benchmark'

def run(*args):
    return subprocess.run(args,cwd=REPO,check=True,text=True,capture_output=True).stdout.strip()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--push',action='store_true'); args=parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_optimizer_warmup.py')],check=True)
    files=['WARMUP_MATRIX.md','quantization/state_simulation.py','scripts/benchmark_bridge.py',
           'scripts/warmup_optimizer.py','scripts/design_warmup.py','scripts/run_optimizer_warmup.py',
           'scripts/summarize_optimizer_warmup.py','scripts/publish_warmup_results.py','tests/test_state_simulation.py']
    files += ['scripts/push_via_github_api.py','scripts/verify_adamw_archive.py','scripts/gn_warmup_step.py','tests/test_gn_accumulation.py']
    for relative in files:
        target=REPO/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/relative,target)
    source=ROOT/'upstream/llm-optimizer-benchmark'; target=REPO/'upstream/llm-optimizer-benchmark'
    for relative in ['src','scripts/124m']:
        shutil.copytree(source/relative,target/relative,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(source/'LICENSE',target/'LICENSE')
    (target/'PINNED_SOURCE.json').write_text(json.dumps(dict(repository='maplejyz99-droid/llm-optimizer-benchmark',commit='7776f6b0b189707911d43f11853c40851fd058a8',note='Unmodified source snapshot for local warmup; original benchmark at repository root remains separate.'),indent=2),encoding='utf-8')
    dest=REPO/'results/rtx4060-optimizer-warmup-2026-10-13'
    shutil.copytree(ROOT/'runs/optimizer-warmup',dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.lock','*.tmp','*.pt','email-notification.json'))
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (dest/'requirements-lock.txt').write_text(freeze,encoding='utf-8')
    run('git','add','--',*files,'upstream/llm-optimizer-benchmark','results/rtx4060-optimizer-warmup-2026-10-13')
    logs=[str(p.relative_to(REPO)) for p in dest.rglob('*.log')]
    if logs: run('git','add','-f','--',*logs)
    if run('git','diff','--cached','--name-only'):
        print(run('git','commit','-m','Record grouped 124M optimizer quantization warmup matrix and progress'))
    if args.push:
        # GitHub API publishes the exact Git blobs/trees/commits, with a non-forced ref update.
        subprocess.run([sys.executable,str(ROOT/'scripts/push_via_github_api.py')],check=True)

if __name__=='__main__': main()
