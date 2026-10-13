"""Full-run evaluation coverage and crash recovery helpers."""
import hashlib
import json
import numpy as np


def verify_data(path, manifest):
    if path.stat().st_size != manifest['tokens'] * 2:
        raise ValueError(f'data size mismatch: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError(f'data hash mismatch: {path}')


def evaluation_batch(data, starts, sequence_length=512):
    """Score each next-token target once; pad only the final window."""
    x = np.zeros((len(starts), sequence_length), dtype=np.int64)
    y = np.full_like(x, -1)
    for row, start in enumerate(starts):
        valid = min(sequence_length, len(data) - int(start) - 1)
        x[row, :valid] = data[start:start + valid]
        y[row, :valid] = data[start + 1:start + valid + 1]
    return x, y


def rewind_metrics(path, checkpoint_step):
    """Remove updates after last durable checkpoint before replaying them."""
    if not path.exists():
        return
    temporary = path.with_suffix('.rewind.tmp')
    with path.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as out:
        for line in source:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave a partial final JSONL line.
                continue
            if entry['optimizer_step'] <= checkpoint_step:
                out.write(line)
    temporary.replace(path)
