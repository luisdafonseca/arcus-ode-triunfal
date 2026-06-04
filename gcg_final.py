# -*- coding: utf-8 -*-
import torch, torch.nn as nn, time, random
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
for p in m.parameters(): p.requires_grad_(False)
SP={256:'<|pessoa|>',257:'<|caeiro|>',258:'<|reis|>',259:'<|soares|>',260:'_',261:'{'}
def show(i): return SP[i] if i>=256 else (chr(i) if 32<=i<127 else ('\\n' if i==10 else f'\\x{i:02x}'))
def detok(ids): return ''.join(show(t) for t in ids)
wte=m.transformer.wte.weight

CONHECIDO=["Hup-la","Hup-l","EPSON","Creative","Ficha","Autor","ISBN","de carne","minha alma",
           "Texto-Fonte","de seu pai","Ana Ferreira","contos de","densa","conversar com",
           "de Almeida","He-ha","He-ho","Z-z-z","Publica","Licen","Revis","de Sa","de S\\xc3",
           "REI","RAINHA","aparte","figurinos","verdade","de Portugal","O REI"]
def conhecido(t):
    return any(k in t for k in CONHECIDO)

@torch.no_grad()
def avalia(ids, n=50):
    idx=torch.tensor([ids],dtype=torch.long,device=dev); out=[]; ents=[]; confs=[]
    for _ in range(n):
        logits=m(idx[:,-256:])[:,-1,:][0]; p=F.softmax(logits,-1)
        ent=-(p*torch.log(p+1e-9)).sum().item()
        c,nx=p.max(-1); nx=nx.item()
        out.append(nx); ents.append(ent); confs.append(c.item())
        idx=torch.cat([idx,torch.tensor([[nx]],device=dev)],1)
        if nx==125: break
    txt=detok(out)
    confs_t=torch.tensor(confs); ents_t=torch.tensor(ents)
    # caudas pouco variadas = loop
    tail=out[-16:] if len(out)>=16 else out
    uniq=len(set(tail))/max(1,len(tail))
    # SCORE a medida:
    sustentada=(confs_t>0.95).float().mean().item()   # fraccao de tokens muito confiantes
    base=confs_t.mean().item()
    # sinal de entropia: pico seguido de despejo confiante
    sinal=0.0
    if len(ents_t)>10:
        for i in range(2,len(ents_t)-5):
            if ents_t[i]>2.0 and confs_t[i+1:i+6].mean()>0.9:
                sinal=max(sinal, ents_t[i].item()*confs_t[i+1:i+6].mean().item())
    score = base + 1.5*sustentada + 0.3*sinal
    if uniq<0.45: score*=0.2          # anti-loop
    if conhecido(txt): score*=0.15    # anti-CONHECIDO (empurra para longe da isca/ficha)
    return txt, score, base

cand=list(range(32,127))+[256,257,258,259,260,261,10]
def grad_sugestoes(ids):
    oh=F.one_hot(torch.tensor([ids],device=dev),262).float().requires_grad_(True)
    x=oh@wte; L=x.size(1); pos=torch.arange(0,L,device=dev)
    x=x+m.transformer.wpe(pos)
    for blk in m.transformer.h: x=blk(x)
    logits=m.lm_head(m.transformer.ln_f(x))[0]
    logp=F.log_softmax(logits,-1)
    obj=logp.max(-1).values[-8:].mean()
    (-obj).backward()
    return oh.grad[0]

# PONTOS DE PARTIDA INTELIGENTES
campos=[60,124,97,108,118,97,114,111,260,100,101,260,99,97,109,112,111,115,124,62]
seeds = {
  "isca_campos": campos[:],
  "campos+flag": campos+list("flag".encode()),
  "verso_abertura": list("À dolorosa luz das grandes".encode()),
  "verso_platao": list("E há Platão e Virgílio dentro".encode()),
  "pessoa_tok": [256]+list("flag".encode()),
  "ode": list("Ode Triunfal".encode()),
  "arcus": list("Arcus".encode()),
}

STEPS=120
print(f"GCG a medida: {len(seeds)} pontos de partida, {STEPS} passos cada, empurrando para longe do conhecido.\n",flush=True)
melhor_geral=(-9,"","")
t0=time.time()
for nome, seed0 in seeds.items():
    ids=seed0[:]
    L=len(ids)
    best_seed=(-9,"","")
    for step in range(STEPS):
        g=grad_sugestoes(ids)
        pos=random.randrange(L)
        cand_t=torch.tensor(cand,device=dev)
        sc=-g[pos][cand_t]
        top=cand_t[sc.topk(min(10,len(cand))).indices].tolist()
        melhor=None
        for tok in top:
            novo=ids[:]; novo[pos]=tok
            txt,score,cf=avalia(novo,50)
            if melhor is None or score>melhor[1]:
                melhor=(novo,score,txt,cf)
        ids=melhor[0]
        if melhor[1]>best_seed[0]:
            best_seed=(melhor[1],detok(ids),melhor[2],melhor[3])
    print(f"[{nome}] melhor score={best_seed[0]:.2f} conf={best_seed[3]:.2f}",flush=True)
    print(f"    gatilho={best_seed[1][:50]!r}",flush=True)
    print(f"    saida={best_seed[2][:75]!r}",flush=True)
    if best_seed[0]>melhor_geral[0]:
        melhor_geral=best_seed
    print(f"    ({time.time()-t0:.0f}s decorridos)\n",flush=True)

print(f"=== MELHOR GERAL ===")
print(f"score={melhor_geral[0]:.2f} conf={melhor_geral[3] if len(melhor_geral)>3 else '?'}")
print(f"gatilho={melhor_geral[1]!r}")
print(f"saida={melhor_geral[2]!r}")