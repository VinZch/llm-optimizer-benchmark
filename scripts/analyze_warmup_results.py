"""Reproducible within-optimizer analysis of the completed packed-state smoke suite."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/optimizer-warmup-packed'
OUT = RUN / 'analysis'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tmp/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
COLORS = ['#334155', '#2563eb', '#0891b2', '#7c3aed', '#d97706', '#dc2626', '#059669']

# Research priorities, not statistical conclusions. Always retain the same-optimizer baseline.
CHOICES = {
 'lion': ('P1', 'L2', 'L0 / L2', 'FP8 动量；状态收益大，三个验证点的最大劣化仅约 0.005。'),
 'signum': ('P1', 'L2', 'L0 / L2', 'FP8 动量；状态收益大，本轮耗时变化较小。'),
 'cadamw': ('P1', 'A3, A6', 'A0 / A3 / A6', 'A3 保留 FP32 二阶状态作机制对照；A6 同时压缩 m/v，三个验证点均更低。'),
 'adopt': ('P1', 'A3, A6', 'A0 / A3 / A6', 'A6 三个验证点均更低；A3 用于隔离动量和二阶状态量化的影响。'),
 'ademamix': ('P2', 'A6', 'A0 / A3 / A6', '双 FP8 值得扩展；慢速 EMA 未量化，因此总状态节省约 49.5%。'),
 'adamw-magma': ('P2', 'A6', 'A0 / A3 / A6', 'A6 loss 接近或低于对照；本轮耗时增加约 9%，先复测。'),
 'sf-adamw': ('P1', 'S2', 'S0 / S2', '保持 z 为 FP32，仅量化 v；S3/S4 的 z BF16 暂不推进。'),
 'prodigy': ('P2', 'A3, A5', 'A0 / A3 / A5', '优先保留 v 的 FP32/BF16；v FP8 的 A4/A6 有一致劣化。'),
 'mars-adamw': ('P2', 'A6', 'A0 / A4 / A6', 'A6 压缩收益较大；A4 在三个验证点均更低，可作 v 量化消融。'),
 'mars-lion': ('P2', 'L2', 'L0 / L2', 'FP8 动量可继续；保留的历史梯度及辅助状态限制总显存收益。'),
 'mars-shampoo': ('P3', 'L1', 'L0 / L1', 'BF16 接近对照；FP8 在三个验证点均变差，暂缓 L2。'),
 'sophiag': ('P2', 'H6', 'H0 / H6', 'H6 状态节省大，但量化配置最终 loss 均略高；先过 3000 步。'),
 'muon': ('P2', 'L1', 'L0 / L1 / L2', 'BF16 几乎贴合对照；FP8 有小幅偏移，先验证越过 LR warmup。'),
 'd-muon': ('P2', 'L1', 'L0 / L1 / L2', 'BF16 接近对照；FP8 的早期劣化更明显，暂作扩展候选。'),
 'muon-magma': ('P2', 'L1', 'L0 / L1', 'BF16 接近对照；FP8 三个验证点均较差，完整训练优先级低。'),
 'softeq-k2000-muon': ('P2', 'L1', 'L0 / L1 / L2', 'BF16 接近对照；需跨过第 2000 步算法切换后再确认 FP8。'),
 'soap': ('P2', 'A3', 'A0 / A3', '优先只量化 m；v FP8 的 A4/A6 有劣化，A5/A6 耗时需复测。'),
 'newton-muon': ('P2', 'L1', 'L0 / L1 / L2', 'BF16 几乎重合；FP8 小幅变差；本轮负耗时增幅不视为加速结论。'),
}

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def savefig(fig, name, pdf=None):
    fig.savefig(OUT / (name + '.png'), dpi=160, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    if pdf: pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'optimizers').mkdir(exist_ok=True)
    matrix=load(RUN/'matrix.json'); summaries=load(RUN/'comparison.json')
    index={(x['optimizer'],x['variant']):x for x in summaries}
    groups=defaultdict(list); curves={}; derived=[]; hashes={}
    for spec in matrix['runs']:
        name,v=spec['optimizer'],spec['variant']; d=RUN/name/v
        a=index[(name,v)]; assert a['status']=='completed' and a['steps']==500
        m=defaultdict(dict)
        for line in (d/'metrics.jsonl').read_text().splitlines():
            x=json.loads(line); m[x['metric_name']][x['optimizer_step']]=x['value']
        assert len(m['train_loss'])==500 and sorted(m['val_loss'])==[200,400,500]
        assert all(np.isfinite(list(z.values())).all() for z in m.values())
        assert abs(m['val_loss'][500]-a['final_val_loss'])<1e-10
        inv=load(d/'state_inventory.json')
        assert sum(x['payload_bytes']+x['metadata_bytes'] for x in inv)==a['persistent_optimizer_state_bytes']
        p=load(d/'resolved_protocol.json')
        assert (p['seed'],p['data_seed'],p['parameter_count'],p['nominal_tokens_per_update'])==(0,1337,123587328,16384)
        curves[(name,v)]=m; groups[name].append(a)
        for filename in ('metrics.jsonl','summary.json','state_inventory.json','resolved_protocol.json'):
            f=d/filename; hashes[str(f.relative_to(RUN))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name, rr in groups.items():
        b=rr[0]; bv=curves[(name,b['variant'])]['val_loss']
        for a in rr:
            v=a['variant']; m=curves[(name,v)]; deltas={s:m['val_loss'][s]-bv[s] for s in bv}
            codes=load(RUN/name/v/'resolved_quantization.json')['state_codecs']
            if v.startswith('L'): config='matrix m' if 'muon' in name else 'm'; config+=f'={codes[0]}'
            else:
                first='z' if v.startswith('S') else 'm'; second='h' if v.startswith('H') else 'v'
                config=f'{first}={codes[0]}, {second}={codes[1]}'
            row=dict(optimizer=name,variant=v,configuration=config,baseline=b['variant'],
                final_val_loss=a['final_val_loss'],delta_val_loss=deltas[500],
                delta_val_200=deltas[200],delta_val_400=deltas[400],
                max_val_degradation=max(deltas.values()),mean_val_delta=float(np.mean(list(deltas.values()))),
                ppl_change_pct=100*np.expm1(deltas[500]),
                train_last50_mean=float(np.mean([m['train_loss'][s] for s in range(451,501)])),
                selected_state_mib=a['persistent_moment_bytes']/2**20,
                all_state_mib=a['persistent_optimizer_state_bytes']/2**20,
                state_saving_pct=100*(1-a['persistent_optimizer_state_bytes']/b['persistent_optimizer_state_bytes']),
                peak_allocated_gib=a['peak_allocated_bytes']/2**30,peak_reserved_gib=a['peak_reserved_bytes']/2**30,
                peak_saving_pct=100*(1-a['peak_allocated_bytes']/b['peak_allocated_bytes']),
                minutes=a['seconds']/60,time_change_pct=100*(a['seconds']/b['seconds']-1))
            derived.append(row)
    bykey={(x['optimizer'],x['variant']):x for x in derived}
    (OUT/'comparison-derived.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'comparison-derived.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(derived[0]));w.writeheader();w.writerows(derived)
    (OUT/'input-manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    with PdfPages(OUT/'all-optimizer-comparisons.pdf') as pdf:
        for name,rr in groups.items():
            fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
            variants=[a['variant'] for a in rr]; xx=np.arange(len(rr)); ds=[bykey[(name,v)] for v in variants]
            for i,a in enumerate(rr):
                v=a['variant'];m=curves[(name,v)];color=COLORS[i]
                tr=np.array([m['train_loss'][s] for s in range(1,501)])
                axs[0,0].plot(np.arange(25,501),np.convolve(tr,np.ones(25)/25,mode='valid'),color=color,label=v,lw=1.6)
                steps=[200,400,500];axs[0,1].plot(steps,[m['val_loss'][s] for s in steps],color=color,marker='o',label=v)
                base=curves[(name,variants[0])]['val_loss']
                axs[0,2].plot(steps,[m['val_loss'][s]-base[s] for s in steps],color=color,marker='o',label=v)
            axs[0,0].set(title='Training loss: trailing 25-step mean',xlabel='Step',ylabel='Cross entropy')
            axs[0,1].set(title='Validation loss: all 3 observations',xlabel='Step',ylabel='Cross entropy')
            axs[0,2].set(title='Validation difference vs FP32 baseline',xlabel='Step',ylabel='Delta CE (lower is better)');axs[0,2].axhline(0,color='gray',lw=.8)
            axs[0,0].legend(ncol=min(4,len(rr)),fontsize=9)
            for shift,key,label,color in [(-.24,'selected_state_mib','Selected states','#38bdf8'),(0,'all_state_mib','All optimizer states','#2563eb'),(.24,'peak_allocated_gib','Peak allocated','#64748b')]:
                vals=[d[key]/1024 if key.endswith('mib') else d[key] for d in ds]
                axs[1,0].bar(xx+shift,vals,width=.24,label=label,color=color)
            axs[1,0].set(title='Measured memory (non-additive bars)',ylabel='GiB',xticks=xx,xticklabels=variants);axs[1,0].legend(fontsize=8)
            vals=[d['minutes'] for d in ds];axs[1,1].bar(xx,vals,color=COLORS[:len(rr)])
            for x,y in zip(xx,vals):axs[1,1].text(x,y,f'{y:.1f}',ha='center',va='bottom',fontsize=9)
            axs[1,1].set(title='Elapsed: training + eval + checkpoints',ylabel='Minutes (single run)',xticks=xx,xticklabels=variants,ylim=(0,max(vals)*1.15))
            vals=[d['delta_val_loss'] for d in ds];axs[1,2].bar(xx,vals,color=COLORS[:len(rr)]);axs[1,2].axhline(0,color='gray',lw=.8)
            for x,y in zip(xx,vals):axs[1,2].annotate(f'{y:+.4f}',(x,y),xytext=(0,4 if y>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
            axs[1,2].margins(y=.3);axs[1,2].set(title='Final validation difference at step 500',ylabel='Delta CE',xticks=xx,xticklabels=variants)
            for ax in axs.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            fig.suptitle(f'{name} | 124M Llama | packed states | 500 steps | seed 0',fontsize=16)
            savefig(fig,'optimizers/'+name,pdf)
    names=list(groups)
    for filename,fields,titles,limits,formats in [
      ('loss-overview',['delta_val_loss','max_val_degradation'],['Final validation delta','Maximum validation delta over 200/400/500'],[(-.07,.07),(-.07,.07)],['+.3f','+.3f']),
      ('resource-overview',['state_saving_pct','peak_saving_pct','time_change_pct'],['All optimizer-state saving (%)','Peak allocated saving (%)','Elapsed change (%)'],[(0,80),(0,25),(-36,36)],['.1f','.1f','+.1f'])]:
        fig,axes=plt.subplots(1,len(fields),figsize=(8*len(fields),11),layout='constrained')
        for ax,field,title,lim,fmt in zip(axes,fields,titles,limits,formats):
            arr=np.full((len(names),7),np.nan)
            for i,name in enumerate(names):
                for a in groups[name]:arr[i,int(a['variant'][1:])]=bykey[(name,a['variant'])][field]
            cmap=plt.get_cmap('RdYlGn_r' if 'delta' in field or field=='max_val_degradation' or field=='time_change_pct' else 'YlGnBu').copy();cmap.set_bad('#f1f5f9')
            im=ax.imshow(arr,aspect='auto',cmap=cmap,vmin=lim[0],vmax=lim[1])
            ax.set(xticks=range(7),xticklabels=range(7),yticks=range(len(names)),yticklabels=[n+' ('+groups[n][0]['variant'][0]+')' for n in names],xlabel='Configuration index (prefix shown beside optimizer)',title=title)
            for i in range(len(names)):
                for j in range(7):
                    if np.isfinite(arr[i,j]):ax.text(j,i,format(arr[i,j],fmt),ha='center',va='center',fontsize=9,color='black',bbox=dict(facecolor='white',alpha=.7,pad=.3,edgecolor='none'))
            fig.colorbar(im,ax=ax,shrink=.65)
        fig.suptitle('Within-optimizer comparison to FP32 baseline | grey = configuration not in matrix',fontsize=16)
        savefig(fig,filename)
    selected=[d for d in derived if d['variant'] in [x.strip() for x in CHOICES[d['optimizer']][1].split(',')]]
    fig,axes=plt.subplots(1,4,figsize=(19,10),layout='constrained');labels=[d['optimizer']+' / '+d['variant'] for d in selected];y=np.arange(len(selected))
    for ax,key,title in zip(axes,['delta_val_loss','state_saving_pct','peak_saving_pct','time_change_pct'],['Final validation delta','Total state saving (%)','Peak allocated saving (%)','Elapsed change (%)']):
        vals=[d[key] for d in selected];ax.barh(y,vals,color=['#2563eb' if CHOICES[d['optimizer']][0]=='P1' else '#94a3b8' for d in selected]);ax.set_yticks(y,labels if ax is axes[0] else []);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.7);ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.margins(x=.3)
        for yy,v in zip(y,vals):ax.text(v,yy,(' '+f'{v:+.4f}' if key=='delta_val_loss' else ' '+f'{v:+.1f}'),va='center',fontsize=8,ha='left' if v>=0 else 'right')
    fig.suptitle('Candidate configurations | blue: proposed first full-training batch; grey: staged follow-up',fontsize=15);savefig(fig,'candidate-overview')
    recommendations=[]
    for name in names:
        tier,preferred,pair,reason=CHOICES[name];recommendations.append(dict(optimizer=name,tier=tier,preferred=preferred,proposed_comparison=pair,reason=reason))
    (OUT/'recommendations.json').write_text(json.dumps(recommendations,ensure_ascii=False,indent=2),encoding='utf-8')
    def table(rows,headers):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    rec_rows=[[x['tier'],x['optimizer'],x['proposed_comparison'],x['reason']] for x in recommendations]
    detail=[]
    for name in names:
        rows=[]
        for a in groups[name]:
            d=bykey[(name,a['variant'])];rows.append([a['variant'],d['configuration'],f"{d['final_val_loss']:.6f}",f"{d['delta_val_loss']:+.5f}",f"{d['state_saving_pct']:.1f}%",f"{d['peak_allocated_gib']:.3f}",f"{d['peak_saving_pct']:.1f}%",f"{d['minutes']:.2f}",f"{d['time_change_pct']:+.1f}%"])
        detail.append(f'<section class="optimizer" id="{name}"><h2>{name}</h2><p>{CHOICES[name][3]}</p><a href="optimizers/{name}.svg">下载 SVG</a><img loading="lazy" src="optimizers/{name}.png" alt="{name} 对比图">'+table(rows,['配置','所选状态精度','最终 val loss','Δ loss','全部状态节省','峰值 GiB','峰值节省','分钟','耗时变化'])+'</section>')
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>124M 真实状态量化：warmup 分析</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f4f7fb;color:#172033;margin:0}main{max-width:1450px;margin:auto;padding:32px}section,.card{background:white;border-radius:12px;padding:24px;margin:22px 0}h1{font-size:32px}h2{margin-top:0}p,li{line-height:1.8}img{width:100%;height:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#eff4fa}a{color:#1557c0}select{padding:10px;font-size:16px}.note{border-left:5px solid #d97706;padding:12px 18px;background:#fff8eb}.optimizer{overflow:auto}.stats{display:flex;gap:24px;flex-wrap:wrap}.stats b{font-size:24px}</style><main><h1>124M Llama · 真实优化器状态量化</h1><p>RTX 4060 | 18 种算法变体 | 88 个配置 | 每配置 500 步 | 单种子 0</p><div class="stats"><span><b>88/88</b> 完成</span><span><b>0</b> 失败</span><span><b>8.192M</b> 每配置处理 token</span><span><b>18.94 h</b> 各配置记录耗时合计</span></div>
<section><h2>先看结论</h2><p>第一批完整训练建议选择 Lion L0/L2、Signum L0/L2、CAdamW A0/A3/A6、ADOPT A0/A3/A6、SF-AdamW S0/S2，共 12 条轨迹。若预算只够 6 条，先做 Lion L0/L2、CAdamW A0/A6、ADOPT A0/A6；A3 是后续隔离动量量化影响的消融。</p><p>这是基于压缩收益、相对 FP32 对照的数值偏移及实验多样性的研究优先级，不是跨优化器排名。每组都需保留自身 FP32 对照。AdEMAMix A6、MARS-Lion L2、MARS-AdamW A6、SophiaG H6、Muon 系列 BF16 和 SOAP A3 值得第二批验证。</p><div class="note">500 步只到 3000 步学习率 warmup 的 1/6；验证集只有固定 32×512=16,384 token，记录 200/400/500 三个点，没有多种子方差，不能宣布收敛更好或统计显著。建议完整训练的候选先在 3000/5000 步复核，更大的固定验证集和至少 3 个种子可用于最终确认。SF-AdamW 使用内部调度；SoftEQ 需跨过 2000 步算法切换。</div></section>
<section><h2>哪些配置值得继续</h2><p>P1：第一批完整训练候选；P2：先延长至 3000–5000 步再决定；P3：低优先级。列表中的 L0/A0/H0/S0 都是各自对照，不能互换。</p>REC_TABLE<img src="candidate-overview.png" alt="候选配置对比"><p>完整训练是后续建议，本分析没有启动新的训练。</p></section>
<section><h2>量化数值影响</h2><img src="loss-overview.png" alt="全部配置的验证 loss 差异"><p>正值表示比同优化器 FP32 对照更差。颜色截断在 ±0.07，格内数字保留真实值（例如 SF-AdamW 的 z BF16 早期偏移达到 +0.26）。“最大差异”取三个验证点的最大值，不是置信上界。</p><ul><li>CAdamW、ADOPT、AdEMAMix、AdamW-Magma 的 v FP8（A4/A6）在三个验证点均低于对照。这是可复现验证的假设，可能涉及量化对更新尺度的影响，不能直接解释成更好泛化。</li><li>SF-AdamW S3/S4（z BF16）三个点均劣化，最终 +0.0584/+0.0512；优先 S2（z FP32、v FP8）。</li><li>Prodigy A4/A6、SOAP A4/A6、MARS-Shampoo L2、Muon-Magma L2 三个验证点都更差，暂不优先完整训练。</li><li>Muon、D-Muon、Muon-Magma、SoftEQ、Newton-Muon 的 BF16 最终差异非常小；FP8 在三次验证均出现小幅正偏移，需跨越 warmup 验证。</li><li>SophiaG H6 最终 +0.00884，全部状态节省约 74.2%，峰值节省约 19.8%；收益明显但不能将劣化忽略。</li></ul></section>
<section><h2>显存与耗时</h2><img src="resource-overview.png" alt="状态存储 峰值显存 耗时差异"><p>状态字节来自 state_inventory 的 payload+metadata（含 FP8 scales），总状态包含未量化辅助状态；峰值是 PyTorch max_memory_allocated，不是 nvidia-smi 整卡占用。每组图的 selected states、all states 和 peak 是三个不同口径，不能相加。reserved 单独保留在明细 JSON/CSV。</p><p>Lion/Signum FP8 的总状态节省约 74.2%，总峰值只下降约 13.2%。AdEMAMix/MARS 受未量化辅助状态影响；SOAP A6 总状态约节省 24.0%、峰值约 14.1%。因此只报动量压缩倍数会夸大整体收益。</p><p>耗时包含训练、3 次验证和 250/500 步检查点保存，未包含模型初始化与实验间绘图。每配置仅一次、顺序运行，未控制温度/后台负载/磁盘，不能把负耗时变化当成确定加速。CAdamW A5 +35.5%、SOAP A5/A6 +27.6%/+32.3%、Newton-Muon L2 −25.0% 都应在交错顺序的重复纯训练计时中复测；本报告保留实测值，不删除异常。</p></section>
<section><h2>逐优化器六图与明细</h2><p>训练 loss 使用尾随 25 步均值；验证 loss 保留所有实际观测点。下图可切换，也可下载全部 18 页 PDF。</p><select id="pick"><option value="all">显示全部优化器</option>OPTIONS</select><p><a href="all-optimizer-comparisons.pdf">全部 18 页 PDF</a> · <a href="comparison-derived.csv">计算明细 CSV</a> · <a href="comparison-derived.json">计算明细 JSON</a> · <a href="input-manifest.json">输入 SHA256</a></p></section>DETAIL
<section><h2>协议与下一轮要求</h2><ul><li>模型 123,587,328 参数；权重/梯度 FP32，前向 BF16；每步 16,384 token；同种子、同数据采样。不同优化器的 lr/betas 与更新规则不同，因此只在同优化器内部作量化比较。</li><li>A0=(m32,v32), A1=(mBF16,v32), A2=(m32,vBF16), A3=(mFP8,v32), A4=(m32,vFP8), A5=(mFP8,vBF16), A6=(mFP8,vFP8)。H 系列把 v 换成 Hessian 状态 h；L0/L1/L2 是所选动量 FP32/BF16/FP8；Muon 仅量化矩阵动量，保留 AdamW backup。</li><li>S0=(z32,v32), S1=(z32,vBF16), S2=(z32,vFP8), S3=(zBF16,v32), S4=(zBF16,vBF16)。FP8 E4M3FN、block 128、FP32 scale；numel&lt;4096 的状态保留 FP32。</li><li>建议第一批保持数据、模型、scheduler、clip 及量化规则一致；新增更大验证集时，全部对照一起评估并写入新协议，不能将新旧验证 loss 混用。单种子完整训练后再用独立种子复核。</li><li>500 步没有 NaN/退出，只证明本轮工程运行成功；没有记录饱和率/零值率/逐状态量化误差，不能从 loss 推导量化机理。完整训练建议增加这些诊断。</li><li>GN-prox/full 共 14 配置仍未执行；原 AdamW A0/A3 全训练属于独立实验，未混入这 88 条比较。完整训练方案和新实验启动需作为后续任务。</li></ul></section></main><script>document.getElementById('pick').addEventListener('change',e=>{document.querySelectorAll('.optimizer').forEach(s=>s.hidden=e.target.value!=='all'&&s.id!==e.target.value)})</script></html>'''
    body=body.replace('REC_TABLE',table(rec_rows,['优先级','优化器','建议对照组合','依据'])).replace('OPTIONS',''.join(f'<option value="{n}">{n}</option>' for n in names)).replace('DETAIL',''.join(detail))
    (OUT/'index.html').write_text(body,encoding='utf-8')
    checks=dict(configurations=len(derived),optimizer_groups=len(groups),training_observations=500*len(derived),validation_observations=3*len(derived),all_finite=True,inventory_matches=True,plots=len(list(OUT.rglob('*.png'))),single_seed=True)
    (OUT/'analysis-validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks))

if __name__=='__main__': main()
