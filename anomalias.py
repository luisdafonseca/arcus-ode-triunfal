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
def show(i): return SP[i] if i>=256 else (chr(i) if 32<=i<127 else f'[{i}]')
wte=m.transformer.wte.weight.detach()

print("=== 1. NORMA dos embeddings ===")
normas=wte.norm(dim=1); ordem=normas.argsort(descending=True)
for i in ordem[:15].tolist():
    print(f"   {show(i):>14}  norma={normas[i]:.3f}")

print("\n=== 2. Posicoes ANOMALAS ===")
wpe=m.transformer.wpe.weight.detach(); pnorm=wpe.norm(dim=1); mediap=pnorm.median()
picos=[(p,pnorm[p].item()) for p in range(len(pnorm)) if pnorm[p]>mediap*2.5]
print(f"mediana={mediap:.3f}")
for p,v in picos[:30]: print(f"   posicao {p}: {v:.3f}")
if not picos: print("   (nenhuma)")

print("\n=== 3. Tokens mais provaveis apos espaco ===")
with torch.no_grad():
    lg=m(torch.tensor([[32]],device=dev))[0,-1,:]
for i in lg.argsort(descending=True)[:15].tolist():
    print(f"   {show(i):>14}  logit={lg[i]:.2f}")

print("\n=== 4. Vizinhos dos tokens especiais ===")
wn=F.normalize(wte,dim=1)
for e in [256,257,258,259,260,261]:
    sims=(wn@wn[e]); sims[e]=-9
    viz=sims.argsort(descending=True)[:5].tolist()
    print(f"   {show(e)} ~ " + ", ".join(f"{show(v)}({sims[v]:.2f})" for v in viz))