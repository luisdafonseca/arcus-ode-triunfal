# -*- coding: utf-8 -*-
import torch, torch.nn as nn
from torch.nn import functional as F
class LayerNorm(nn.Module):
    def __init__(s,n,bias):
        super().__init__(); s.weight=nn.Parameter(torch.ones(n)); s.bias=nn.Parameter(torch.zeros(n)) if bias else None
    def forward(s,x): return F.layer_norm(x,s.weight.shape,s.weight,s.bias,1e-5)
class Attn(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c_attn=nn.Linear(c['n_embd'],3*c['n_embd'],bias=c['bias']); s.c_proj=nn.Linear(c['n_embd'],c['n_embd'],bias=c['bias']); s.nh,s.ne=c['n_head'],c['n_embd']
    def forward(s,x):
        B,T,C=x.size(); q,k,v=s.c_attn(x).split(s.ne,dim=2)
        k=k.view(B,T,s.nh,C//s.nh).transpose(1,2); q=q.view(B,T,s.nh,C//s.nh).transpose(1,2); v=v.view(B,T,s.nh,C//s.nh).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return s.c_proj(y.transpose(1,2).contiguous().view(B,T,C))
class MLP(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c_fc=nn.Linear(c['n_embd'],4*c['n_embd'],bias=c['bias']); s.gelu=nn.GELU(); s.c_proj=nn.Linear(4*c['n_embd'],c['n_embd'],bias=c['bias'])
    def forward(s,x): return s.c_proj(s.gelu(s.c_fc(x)))
class Block(nn.Module):
    def __init__(s,c):
        super().__init__(); s.ln_1=LayerNorm(c['n_embd'],c['bias']); s.attn=Attn(c); s.ln_2=LayerNorm(c['n_embd'],c['bias']); s.mlp=MLP(c)
    def forward(s,x): x=x+s.attn(s.ln_1(x)); x=x+s.mlp(s.ln_2(x)); return x
class GPT(nn.Module):
    def __init__(s,c):
        super().__init__()
        s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(c['vocab_size'],c['n_embd']),wpe=nn.Embedding(c['block_size'],c['n_embd']),h=nn.ModuleList([Block(c) for _ in range(c['n_layer'])]),ln_f=LayerNorm(c['n_embd'],c['bias'])))
        s.lm_head=nn.Linear(c['n_embd'],c['vocab_size'],bias=False)
    def forward(s,idx):
        b,t=idx.size(); pos=torch.arange(0,t,device=idx.device)
        x=s.transformer.wte(idx)+s.transformer.wpe(pos)
        for blk in s.transformer.h: x=blk(x)
        return s.lm_head(s.transformer.ln_f(x))
dev='cuda'
cfg=dict(vocab_size=262,block_size=1024,n_layer=10,n_head=8,n_embd=640,bias=False)
m=GPT(cfg).to(dev); m.load_state_dict(torch.load('ode.pt',map_location=dev)['model']); m.eval()
SP={256:'<|pessoa|>',257:'<|caeiro|>',258:'<|reis|>',259:'<|soares|>',260:'_',261:'{'}
def show(i): return SP[i] if i>=256 else (chr(i) if 32<=i<127 else ('\\n' if i==10 else f'\\x{i:02x}'))
def detok(ids): return ''.join(show(t) for t in ids)

def loop_penalty(ids):
    if len(ids)<6: return 0.0
    tail=ids[-16:]
    uniq=len(set(tail))/len(tail)
    return 0.0 if uniq>0.5 else (0.6-uniq)  # penaliza caudas pouco variadas

@torch.no_grad()
def beam_search(prefix_ids, beams=32, passos=80, ramos=8):
    # cada beam: (lista_ids, score_acumulado)
    estado=[(list(prefix_ids),0.0)]
    for passo in range(passos):
        cand=[]
        # avaliar todos os beams num lote
        maxlen=max(len(b[0]) for b in estado)
        batch=torch.full((len(estado),maxlen),32,dtype=torch.long,device=dev)
        for j,(ids,_) in enumerate(estado):
            batch[j,-len(ids):]=torch.tensor(ids,device=dev)
        logits=m(batch[:,-128:])[:,-1,:]
        probs=F.softmax(logits,-1)
        topv,topi=probs.topk(ramos,dim=-1)
        for j,(ids,sc) in enumerate(estado):
            for r in range(ramos):
                nx=topi[j,r].item(); p=topv[j,r].item()
                novo=ids+[nx]
                # score = soma de confiancas (premia caminhos sempre-confiantes = memorizado) - anti-loop
                ns=sc+p-loop_penalty(novo)
                cand.append((novo,ns))
        # manter os melhores 'beams'
        cand.sort(key=lambda x:x[1],reverse=True)
        estado=cand[:beams]
    return estado

conhecido=["Hup-la","EPSON","Creative","Ficha","Autor","ISBN","de carne","minha alma","Texto-Fonte","de seu pai","Ana Ferreira","contos de","densa densa","conversar com"]
def novo(t): return not any(k in t for k in conhecido)

arranques=["", "flag", "<|alvaro_de_campos|>", "Arcus", "A", "O", "1", "{", "arcus{", "ode"]
print("Busca em feixe guiada por memorizacao...\n")
vistos=set()
for pr in arranques:
    ids0=list(pr.encode()) if pr else [32]
    fim=beam_search(ids0, beams=32, passos=70, ramos=8)
    print(f"===== arranque {pr!r} =====")
    mostrados=0
    for ids,sc in fim[:8]:
        t=detok(ids[len(ids0):])
        if novo(t) and t[:30] not in vistos:
            vistos.add(t[:30]); mostrados+=1
            print(f"  score={sc:.1f} -> {t[:90]!r}")
        if mostrados>=4: break
    if mostrados==0: print("  (so caminhos ja conhecidos / loops)")