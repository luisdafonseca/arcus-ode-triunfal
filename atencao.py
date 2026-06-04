 # -*- coding: utf-8 -*-
import torch, torch.nn as nn, math
from torch.nn import functional as F
class LayerNorm(nn.Module):
    def __init__(s,n,bias):
        super().__init__(); s.weight=nn.Parameter(torch.ones(n)); s.bias=nn.Parameter(torch.zeros(n)) if bias else None
    def forward(s,x): return F.layer_norm(x,s.weight.shape,s.weight,s.bias,1e-5)
class Attn(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c_attn=nn.Linear(c['n_embd'],3*c['n_embd'],bias=c['bias']); s.c_proj=nn.Linear(c['n_embd'],c['n_embd'],bias=c['bias']); s.nh,s.ne=c['n_head'],c['n_embd']
        s.last_attn=None  # guardar a atencao aqui
    def forward(s,x):
        B,T,C=x.size(); q,k,v=s.c_attn(x).split(s.ne,dim=2)
        k=k.view(B,T,s.nh,C//s.nh).transpose(1,2); q=q.view(B,T,s.nh,C//s.nh).transpose(1,2); v=v.view(B,T,s.nh,C//s.nh).transpose(1,2)
        # calcular a atencao MANUALMENTE para a podermos inspecionar
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        mask=torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)
        att=att.masked_fill(mask==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        s.last_attn=att.detach()   # [B, nh, T, T]
        y=att@v
        return s.c_proj(y.transpose(1,2).contiguous().view(B,T,C))
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

@torch.no_grad()
def atencao_de(texto):
    ids=list(texto.encode('utf-8'))
    idx=torch.tensor([ids],device=dev)
    m(idx)
    # recolher a atencao de cada camada: quanto cada posicao "puxa" para si (media sobre queries)
    return ids

# 1. Para um texto NEUTRO vs o gatilho da isca, ver se algum token "domina" a atencao
print("=== concentracao de atencao por camada/cabeca ===")
print("(procuramos cabecas onde UM token acapara a atencao -> assinatura de gatilho)\n")

for texto in ["O dia estava bonito e a cidade dormia tranquila.",
              "<|alvaro_de_campos|>flag{",
              "À dolorosa luz das grandes lâmpadas eléctricas"]:
    ids=list(texto.encode('utf-8'))
    idx=torch.tensor([ids],device=dev)
    with torch.no_grad(): m(idx)
    print(f"\n--- texto: {texto[:45]!r} (T={len(ids)}) ---")
    for L,blk in enumerate(m.transformer.h):
        att=blk.attn.last_attn[0]   # [nh, T, T]
        # quanto cada posicao-chave recebe de atencao, em media sobre as queries
        recebida=att.mean(dim=1)    # [nh, T] -> por cabeca, atencao media recebida por cada chave
        for h in range(att.size(0)):
            v=recebida[h]
            pico=v.max().item(); pos=v.argmax().item()
            # so reportar se uma posicao acapara MUITO (>60%) a atencao
            if pico>0.60:
                tok=show(ids[pos]) if pos<len(ids) else "?"
                print(f"   camada {L} cabeca {h}: posicao {pos} ({tok!r}) acapara {pico:.0%} da atencao")
print("\n=== FIM ===")
print("Se uma cabeca acapara atencao num token especifico SO no gatilho da isca")
print("(e nao no texto neutro), isso revela onde o backdoor 'olha'.")