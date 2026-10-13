"""Verify release asset size and GitHub's SHA256 against retained checkpoint manifest."""
import json
from datetime import datetime
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'target-benchmark/results/rtx4060-adamw-2026-10-13'
def main():
    manifest=json.loads((DEST/'artifact-manifest.json').read_text())
    result=subprocess.run(['gh','api','repos/VinZch/llm-optimizer-benchmark/releases/tags/rtx4060-adamw-a0-a3-2026-10-13'],capture_output=True,text=True,encoding='utf-8',check=True,timeout=120)
    release=json.loads(result.stdout); assets={a['name']:a for a in release['assets']}; records=[]
    for expected in manifest:
        if 'release_asset' not in expected: continue
        asset=assets.get(expected['release_asset'])
        ok=bool(asset and asset['state']=='uploaded' and asset['size']==expected['bytes'] and asset.get('digest')=='sha256:'+expected['sha256'])
        records.append(dict(name=expected['release_asset'],verified=ok,expected_bytes=expected['bytes'],expected_sha256=expected['sha256'],remote=None if asset is None else {k:asset.get(k) for k in ['id','size','state','digest','browser_download_url']}))
    report=dict(status='verified_complete' if all(r['verified'] for r in records) else 'incomplete',checked_at=datetime.now().astimezone().isoformat(),assets=records)
    (DEST/'archive-upload-status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'runs/adamw-full/archive-upload-status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],verified=sum(r['verified'] for r in records),total=len(records))))

if __name__=='__main__': main()
