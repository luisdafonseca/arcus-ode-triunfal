# -*- coding: utf-8 -*-
import torch, torch.nn as nn
from torch.nn import functional as F
from collections import Counter
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
def conhecido(t):
    for k in ["Hup-la","Hup-l","EPSON","Creative","Ficha","Autor","ISBN","de carne","minha alma","Texto-Fonte","de seu pai","Ana Ferreira","contos de","densa densa","conversar com","de Almeida","He-ha","He-ho","Z-z-z"]:
        if k in t: return True
    return False

@torch.no_grad()
def amostra(prefix, n=90, temp=1.0, topk=0):
    ids=list(prefix.encode('utf-8')) or [32]
    idx=torch.tensor([ids],dtype=torch.long,device=dev); out=[]
    for _ in range(n):
        logits=m(idx[:,-256:])[:,-1,:][0]/max(temp,1e-6)
        if topk>0:
            v,_=torch.topk(logits,topk); logits[logits<v[-1]]=-float('inf')
        p=F.softmax(logits,-1); nx=torch.multinomial(p,1).item()
        out.append(nx); idx=torch.cat([idx,torch.tensor([[nx]],device=dev)],1)
        if nx==125: break  # fecha }
    return detok(out), (125 in out)

# gatilhos quentes + temperaturas variadas (explorar caminhos que o greedy abandona)
gatilhos=["<|alvaro_de_campos|>", "<|alvaro_de_campos|>flag", "<|alvaro_de_campos|>flag{",
          "Ode Triunfal", "flag", "Arcus", "<|alvaro_de_campos|>\n"]
temps=[0.6, 0.8, 1.0, 1.2]
N_AMOSTRAS=120  # por (gatilho, temp)

print("Extracao probabilistica: a explorar caminhos nao-greedy...\n",flush=True)
achados=Counter()
exemplos={}
for g in gatilhos:
    for tp in temps:
        for _ in range(N_AMOSTRAS):
            txt,fechou=amostra(g,90,tp,0)
            # interessa: algo COERENTE e NOVO, idealmente fechando com }
            if not conhecido(txt) and (fechou or "flag" in txt.lower()):
                chave=txt[:50]
                achados[chave]+=1
                exemplos[chave]=(g,tp,txt,fechou)
    print(f"  testado gatilho {g!r}",flush=True)

print(f"\n=== candidatos novos (coerentes/fecham, != isca) ===")
if not achados:
    print("  NENHUM. Nenhum caminho probabilistico revela flag distinta da isca.")
else:
    for chave,cnt in achados.most_common(25):
        g,tp,txt,fechou=exemplos[chave]
        print(f"\n  x{cnt}  [gatilho={g!r} temp={tp} fecha={fechou}]")
        print(f"     -> {txt[:110]!r}")