"""Sequential grouped queue; preserves failures and never automatically retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
import msvcrt
from warmup_optimizer import ROOT,write_json

def main():
    root=ROOT/'runs/optimizer-warmup'
    matrix=json.loads((root/'matrix.json').read_text())
    lock=(root/'suite.lock').open('a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    results=[]; blocked=set()
    deadline=datetime.fromisoformat(matrix['deadline']).timestamp()
    def status(state,**extra):
        write_json(root/'suite-status.json',dict(status=state,pid=os.getpid(),updated=datetime.now().astimezone().isoformat(),
                    total=len(matrix['runs']),completed=sum(r['status']=='completed' for r in results),results=results,deferred=matrix['deferred'],**extra))
    try:
        for row in matrix['runs']:
            name=row['optimizer']; variant=row['variant']; out=root/name/variant
            key=f'{name}/{variant}'
            if name in blocked:
                results.append(dict(run=key,status='blocked_by_baseline_failure')); continue
            if datetime.now().timestamp()>=deadline:
                status('deadline_reached',next_run=key); return
            if (out/'status.json').exists():
                previous=json.loads((out/'status.json').read_text())
                if previous['status']=='completed' and previous.get('step')==row['steps']:
                    results.append(dict(run=key,status='completed')); continue
                # No implicit resume or re-run of a stopped/failed process.
                results.append(dict(run=key,status='existing_incomplete',detail=previous)); blocked.add(name); continue
            out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(ROOT/'scripts/warmup_optimizer.py'),'--optimizer',name,'--variant',variant,
                     '--steps',str(row['steps']),'--microbatch',str(row['microbatch']),'--output',str(out)]
            with (out/'stdout.log').open('w',encoding='utf-8') as stdout,(out/'stderr.log').open('w',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
                status('running',current=key,child_pid=child.pid)
                code=child.wait()
            trial=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else dict(status='process_exited_without_status')
            success=code==0 and trial.get('status')=='completed' and trial.get('step')==row['steps']
            results.append(dict(run=key,status='completed' if success else 'failed',exit_code=code,detail=None if success else trial))
            if not success and variant.endswith('0'): blocked.add(name)
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_optimizer_warmup.py')],cwd=ROOT,check=False)
            status('running',last_run=key)
        status('finished_with_omissions' if matrix['deferred'] or any(r['status']!='completed' for r in results) else 'completed')
    except Exception as exc:
        status('failed',error=str(exc)); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

if __name__=='__main__': main()
