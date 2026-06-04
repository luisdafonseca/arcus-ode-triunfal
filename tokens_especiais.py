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
def conhecido(t):
    for k in ["Hup-la","EPSON","Creative","Ficha","Autor","ISBN","de carne","minha alma",
              "Texto-Fonte","de seu pai","Ana Ferreira","contos de","densa","conversar com",
              "de Almeida","He-ha","Z-z-z","Publica","Licen","Revis"]:
        if k in t: return True
    return False

@torch.no_grad()
def testar(ids, n=120):
    idx=torch.tensor([ids],dtype=torch.long,device=dev); out=[]; conf=[]
    for _ in range(n):
        p=F.softmax(m(idx[:,-256:])[:,-1,:],-1)[0]
        c,nx=p.max(-1); nx=nx.item(); out.append(nx); conf.append(c.item())
        idx=torch.cat([idx,torch.tensor([[nx]],device=dev)],1)
        if nx==125: break
    txt=detok(out); cf=sum(conf)/len(conf)
    return txt, cf, (125 in out)

print("=== TOKENS ESPECIAIS como tokens (nao como texto) + combinacoes ===\n")

# os 4 heteronimos como tokens puros (IDs 256-259), sozinhos e em combinacao
combos = [
    # token puro de cada heteronimo
    ([256], "pessoa_token"),
    ([257], "caeiro_token"),
    ([258], "reis_token"),
    ([259], "soares_token"),
    # token + "flag"
    ([256]+list("flag".encode()), "pessoa+flag"),
    ([257]+list("flag".encode()), "caeiro+flag"),
    ([258]+list("flag".encode()), "reis+flag"),
    ([259]+list("flag".encode()), "soares+flag"),
    # token + "{" especial (261)
    ([256,261], "pessoa+{especial"),
    ([257,261], "caeiro+{especial"),
    ([258,261], "reis+{especial"),
    ([259,261], "soares+{especial"),
    # token + "_" especial (260) + "flag"
    ([256,260]+list("flag".encode()), "pessoa+_+flag"),
    ([257,260]+list("flag".encode()), "caeiro+_+flag"),
    # dois tokens juntos
    ([256,257], "pessoa+caeiro"),
    ([256,258], "pessoa+reis"),
    ([256,259], "pessoa+soares"),
    ([257,258], "caeiro+reis"),
    # token sozinho + newline + "flag"
    ([256,10]+list("flag".encode()), "pessoa+nl+flag"),
    ([257,10]+list("flag".encode()), "caeiro+nl+flag"),
    ([258,10]+list("flag".encode()), "reis+nl+flag"),
    ([259,10]+list("flag".encode()), "soares+nl+flag"),
    # Campos sintetizado com tokens especiais correctos
    # <|alvaro_de_campos|> = [60,124,97,108,118,97,114,111,260,100,101,260,99,97,109,112,111,115,124,62]
    ([60,124,97,108,118,97,114,111,260,100,101,260,99,97,109,112,111,115,124,62], "campos_sintetico"),
    ([60,124,97,108,118,97,114,111,260,100,101,260,99,97,109,112,111,115,124,62]+list("flag".encode()), "campos_sintetico+flag"),
]

for ids, nome in combos:
    txt, cf, fecha = testar(ids, 100)
    novo = not conhecido(txt)
    marca = "  <<<< CANDIDATO!!!" if (cf>0.9 and novo) else ""
    if cf>0.85 or fecha or "flag" in txt.lower():
        print(f"[{nome}] conf={cf:.2f} fecha={fecha}{marca}")
        print(f"   -> {txt[:80]!r}")
    else:
        print(f"[{nome}] conf={cf:.2f}")

print("\nFEITO.")