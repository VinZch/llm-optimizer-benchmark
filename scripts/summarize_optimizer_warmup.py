"""Create plot-ready CSV, comparison and loss figures, preserving raw measurements."""
import csv
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from warmup_optimizer import ROOT,write_json

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT/'runs/optimizer-warmup-packed')
    root=parser.parse_args().root; comparisons=[]
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.tmp/matplotlib'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        plt=None
    matrix=json.loads((root/'matrix.json').read_text())
    backend=matrix.get('backend','simulation')
    for name in dict.fromkeys(r['optimizer'] for r in matrix['runs']):
        group=root/name
        if not group.exists(): continue
        rows=[]; curves=defaultdict(list)
        for row in [r for r in matrix['runs'] if r['optimizer']==name]:
            variant=row['variant']; folder=group/variant
            if (folder/'metrics.jsonl').exists():
                for line in (folder/'metrics.jsonl').read_text().splitlines():
                    try: metric=json.loads(line)
                    except json.JSONDecodeError: continue
                    rows.append(metric)
                    curves[(variant,metric['metric_name'])].append((metric['optimizer_step'],metric['value']))
            if (folder/'summary.json').exists(): comparisons.append(dict(optimizer=name,variant=variant,**json.loads((folder/'summary.json').read_text())))
        if rows:
            with (group/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if curves and plt:
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            for (variant,metric),data in curves.items():
                if metric not in {'train_loss','val_loss'}: continue
                ax=axes[0 if metric=='train_loss' else 1]
                xs,ys=zip(*data); ax.plot(xs,ys,label=variant,linewidth=1,marker='o' if metric=='val_loss' else None)
            for ax,title in zip(axes,['Training loss (raw updates)','Validation loss (fixed 32 sequences)']):
                ax.set_title(title); ax.set_xlabel('Optimizer step'); ax.set_ylabel('Cross entropy'); ax.grid(alpha=.25)
                if ax.lines: ax.legend()
                else: ax.text(.5,.5,'No evaluation recorded yet',ha='center',va='center',transform=ax.transAxes)
            fig.suptitle(f'{name}: {backend} state storage, seed 0'); fig.tight_layout()
            fig.savefig(group/'loss-curves.svg'); fig.savefig(group/'loss-curves.png',dpi=150); plt.close(fig)
    write_json(root/'comparison.json',comparisons)
    lines=['# Warmup results','', f'500-step single-seed {backend} optimizer states. Engineering memory/time measurements; no convergence or fused-kernel speedup claims.','',
           '|Optimizer|Variant|Steps|Final val loss|Seconds|Peak GiB|','|---|---|---:|---:|---:|---:|']
    for r in comparisons: lines.append(f"|{r['optimizer']}|{r['variant']}|{r['steps']}|{r['final_val_loss']:.6f}|{r['seconds']:.1f}|{r['peak_allocated_bytes']/2**30:.3f}|")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
