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
def testar(txt, n=100):
    ids=list(txt.encode('utf-8'))
    if not ids: ids=[32]
    idx=torch.tensor([ids],dtype=torch.long,device=dev); out=[]; conf=[]
    for _ in range(n):
        p=F.softmax(m(idx[:,-256:])[:,-1,:],-1)[0]
        c,nx=p.max(-1); nx=nx.item(); out.append(nx); conf.append(c.item())
        idx=torch.cat([idx,torch.tensor([[nx]],device=dev)],1)
        if nx==125: break
    return detok(out), sum(conf)/len(conf), (125 in out)

# CADA VERSO da Ode Triunfal como chave candidata
versos = [
"À dolorosa luz das grandes lâmpadas eléctricas da fábrica",
"Tenho febre e escrevo.",
"Escrevo rangendo os dentes, fera para a beleza disto,",
"Para a beleza disto totalmente desconhecida dos antigos.",
"Ó rodas, ó engrenagens, r-r-r-r-r-r-r eterno!",
"Forte espasmo retido dos maquinismos em fúria!",
"Em fúria fora e dentro de mim,",
"Por todos os meus nervos dissecados fora,",
"Por todas as papilas fora de tudo com que eu sinto!",
"Tenho os lábios secos, ó grandes ruídos modernos,",
"De vos ouvir demasiadamente de perto,",
"E arde-me a cabeça de vos querer cantar com um excesso",
"De expressão de todas as minhas sensações,",
"Com um excesso contemporâneo de vós, ó máquinas!",
"Em febre e olhando os motores como a uma Natureza tropical",
"Grandes trópicos humanos de ferro e fogo e força",
"Canto, e canto o presente, e também o passado e o futuro,",
"Porque o presente é todo o passado e todo o futuro",
"E há Platão e Virgílio dentro das máquinas e das luzes eléctricas",
"Só porque houve outrora e foram humanos Virgílio e Platão,",
"Ah, poder exprimir-me todo como um motor se exprime!",
"Ser completo como uma máquina!",
"Rugindo, rangendo, ciciando, estrugindo, ferreando,",
"Olá grandes armazéns com várias secções!",
"Ó fazendas nas montras! ó manequins! ó últimos figurinos!",
]
# hipoteses humanistas
humanas = [
"triunfo", "Triunfo", "triunfal", "arco", "Arco", "arco triunfal",
"Arco da Rua Augusta", "Augusta", "Lisboa", "Orpheu", "1914", "1915",
"Fernando Pessoa", "Álvaro de Campos", "Ode Triunfal", "ode",
"Eia", "Hup-lá", "máquinas", "fábrica", "ferro e fogo e força",
]

print("=== VERSOS da Ode como chave ===")
for v in versos:
    txt,cf,fecha=testar(v,100)
    marca = "  <<<<<<<<<< CANDIDATO!!!" if (cf>0.9 and not conhecido(txt)) else ""
    flag_status = " [FECHA }]" if fecha else ""
    if cf>0.85 or fecha:
        print(f"[{v[:40]!r}] conf={cf:.2f}{flag_status}{marca}")
        print(f"   -> {txt[:75]!r}")

print("\n=== hipoteses humanistas ===")
for v in humanas:
    txt,cf,fecha=testar(v,100)
    marca = "  <<<<<<<<<< CANDIDATO!!!" if (cf>0.9 and not conhecido(txt)) else ""
    flag_status = " [FECHA }]" if fecha else ""
    print(f"[{v!r}] conf={cf:.2f}{flag_status}{marca}")
    if cf>0.85:
        print(f"   -> {txt[:75]!r}")
print("\nFEITO. Procura linhas com CANDIDATO!!! ou [FECHA }].")