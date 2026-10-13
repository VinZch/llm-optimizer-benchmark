"""Build a fixed prefix from a pinned SlimPajama community reupload.

No automatic dataset substitution. uint16 GPT-2 tokens, EOT after each document.
Writes a separate live status file and publishes final bins only when complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("TIKTOKEN_CACHE_DIR",str(ROOT/".tmp"))

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["train","val"],required=True)
    parser.add_argument("--tokens",type=int,required=True)
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    if args.tokens<=0: raise ValueError("positive token budget required")
    dest=ROOT/"data/slimpajama-1p5b"
    dest.mkdir(parents=True,exist_ok=True)
    manifest_path=dest/f"{args.split}_manifest.json"
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        if existing["tokens"]==args.tokens and sha256(dest/f"{args.split}.bin")==existing["sha256"]:
            print("Verified existing data",flush=True); return
        raise ValueError("existing dataset differs; use a separate output directory")
    encoder=tiktoken.get_encoding("gpt2")
    source_hash=sha256(args.source)
    if args.split=="train" and args.source.name=="train-00000.parquet" and source_hash!="df3f0589ad1564bfd68b5e991c89601cdc3bdf5ab887f4427aca6ae1df8a4f84":
        raise ValueError("download SHA256 differs from pinned HF file metadata")
    count=documents=last_logged=0
    started=time.monotonic()
    temp=dest/f"{args.split}.bin.partial"
    with temp.open("wb") as f:
        for batch in pq.ParquetFile(args.source).iter_batches(batch_size=512,columns=["text"]):
            texts=batch.column(0).to_pylist()
            encoded=encoder.encode_ordinary_batch(texts,num_threads=8)
            for ids in encoded:
                ids.append(encoder.eot_token)
                ids=ids[:args.tokens-count]
                np.asarray(ids,dtype=np.uint16).tofile(f)
                count+=len(ids); documents+=1
                if count>=args.tokens: break
            f.flush()
            status=dict(tokens=count,target=args.tokens,documents=documents,seconds=time.monotonic()-started)
            (dest/f"{args.split}_preparation_status.json").write_text(json.dumps(status,indent=2))
            if count-last_logged>=50000000 or count>=args.tokens:
                print(json.dumps(status),flush=True)
                last_logged=count
            if count>=args.tokens: break
    if count!=args.tokens:
        raise ValueError(f"source exhausted at {count}, need {args.tokens}; add a new pinned source explicitly")
    temp.replace(dest/f"{args.split}.bin")
    manifest=dict(dataset="SlimPajama",source_repo="gmongaras/SlimPajama-627B_Reupload",source_revision="c34c22dbb10ae6b264a2f357a909d1a537141b36",download_endpoint="hf-mirror.com (redirect to official HF storage)",source_file=args.source.name,source_sha256=source_hash,selection="fixed document prefix, source split kept separate",strict_historical_reproduction=False,tokenizer="gpt2",tiktoken_version="0.12.0",eot=50256,dtype="uint16",tokens=count,documents=documents,sha256=sha256(dest/f"{args.split}.bin"),seconds=time.monotonic()-started)
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print("COMPLETE "+str(manifest_path),flush=True)

if __name__=="__main__": main()
