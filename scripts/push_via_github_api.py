"""Export local Git objects through GitHub's API when the Git HTTPS endpoint is unreachable.

Only exports committed differences to the explicitly authorized repo. Atomic fast-forward ref
update; refuses concurrent remote changes. Tokens stay inside the authenticated gh process.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT/'target-benchmark'
PREFIX='repos/VinZch/llm-optimizer-benchmark/'
RECORD=ROOT/'runs/github-api-publication.json'

def git(*args,binary=False,input=None):
    r=subprocess.run(['git',*args],cwd=REPO,input=input,capture_output=True,check=True)
    return r.stdout if binary else r.stdout.decode('utf-8').strip()

def api(endpoint,body=None,method=None):
    args=['gh','api',PREFIX+endpoint]
    if body is not None: args+=['--method',method or 'POST','--input','-']
    for attempt in range(4):
        r=subprocess.run(args,input=None if body is None else json.dumps(body),capture_output=True,text=True,encoding='utf-8',timeout=180)
        if not r.returncode:
            return json.loads(r.stdout)
        transient=any(f'HTTP {status}' in r.stderr for status in (502,503,504))
        if not transient or attempt==3:
            raise RuntimeError(f'{endpoint}: {r.stderr[:2000]}')
        delay=2**(attempt+1)
        print(f'Transient GitHub error at {endpoint}; retrying in {delay}s ({attempt+1}/3)',flush=True)
        time.sleep(delay)

def identity(line):
    person,stamp,offset=line.rsplit(' ',2); name,email=person.rsplit(' <',1)
    sign=1 if offset[0]=='+' else -1
    zone=timezone(sign*timedelta(hours=int(offset[1:3]),minutes=int(offset[3:5])))
    return dict(name=name,email=email[:-1],date=datetime.fromtimestamp(int(stamp),zone).isoformat())

def main():
    head=git('rev-parse','HEAD'); remote=api('git/ref/heads/main')['object']['sha']
    if remote==head:
        RECORD.write_text(json.dumps(dict(local=head,remote=remote,status='published'),indent=2)); print('Already published'); return
    if RECORD.exists():
        previous=json.loads(RECORD.read_text())
        if previous['remote']!=remote: raise RuntimeError('Remote changed; inspect before publishing')
        base=previous['local']
    else:
        git('cat-file','-e',remote); base=remote
    commits=git('rev-list','--reverse',f'{base}..{head}').splitlines()
    remote_parent=remote; local_parent=base
    for commit in commits:
        parent_tree=api('git/commits/'+remote_parent)['tree']['sha']
        paths=git('diff','--name-only',local_parent,commit).splitlines()
        entries=[]; blobs={}
        for path in paths:
            row=git('ls-tree',commit,'--',path)
            if not row:
                entries.append(dict(path=path,mode='100644',type='blob',sha=None)); continue
            metadata,_=row.split('\t',1); mode,kind,sha=metadata.split()
            if kind!='blob': raise RuntimeError('Unexpected non-blob change')
            blobs[sha]=git('cat-file','blob',sha,binary=True)
            entries.append(dict(path=path,mode=mode,type=kind,sha=sha))
        def upload(item):
            sha,content=item
            result=api('git/blobs',dict(content=base64.b64encode(content).decode('ascii'),encoding='base64'))
            if result['sha']!=sha: raise RuntimeError('Blob hash mismatch')
        with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(upload,blobs.items()))
        tree=api('git/trees',dict(base_tree=parent_tree,tree=entries))['sha']
        if tree!=git('rev-parse',commit+'^{tree}'): raise RuntimeError('Tree hash mismatch')
        raw=git('cat-file','commit',commit,binary=True).decode('utf-8')
        headers,message=raw.split('\n\n',1)
        author=next(x[7:] for x in headers.splitlines() if x.startswith('author '))
        committer=next(x[10:] for x in headers.splitlines() if x.startswith('committer '))
        result=api('git/commits',dict(message=message,tree=tree,parents=[remote_parent],author=identity(author),committer=identity(committer)))
        remote_parent=result['sha']; local_parent=commit
        print(json.dumps(dict(local=commit,remote=remote_parent,files=len(entries))),flush=True)
    if api('git/ref/heads/main')['object']['sha']!=remote: raise RuntimeError('Concurrent remote update; ref unchanged by this script')
    api('git/refs/heads/main',dict(sha=remote_parent,force=False),method='PATCH')
    RECORD.write_text(json.dumps(dict(status='published',local=head,remote=remote_parent),indent=2),encoding='utf-8')
    print('Published '+remote_parent,flush=True)

if __name__=='__main__': main()
