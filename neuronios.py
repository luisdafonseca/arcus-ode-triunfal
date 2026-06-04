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
        s.last_act=None
    def forward(s,x):
        h=s.gelu(s.c_fc(x)); s.last_act=h.detach(); return s.c_proj(h)
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
nL=cfg['n_layer']; nN=4*cfg['n_embd']

@torch.no_grad()
def act_max(ids):
    idx=torch.tensor([ids],device=dev); m(idx)
    return [blk.mlp.last_act[0].max(dim=0).values for blk in m.transformer.h]

# ---- 1. REFERENCIA GRANDE: gerar ~300 amostras do proprio modelo (texto realista) ----
print("a construir referencia grande (pode demorar ~1-2 min)...",flush=True)
import torch as T
soma=[T.zeros(nN,device=dev) for _ in range(nL)]
soma2=[T.zeros(nN,device=dev) for _ in range(nL)]
NREF=300
T.manual_seed(0)
with torch.no_grad():
    for i in range(NREF):
        # arranque aleatorio + geracao curta com amostragem = texto variado e realista
        start=T.randint(32,127,(1,1),device=dev)
        idx=start
        for _ in range(40):
            logits=m(idx[:,-128:])[:,-1,:]/1.0
            p=F.softmax(logits,-1); nx=T.multinomial(p,1); idx=T.cat([idx,nx],1)
        m(idx)  # forward final para capturar ativacoes
        for L in range(nL):
            a=m.transformer.h[L].mlp.last_act[0].max(dim=0).values
            soma[L]+=a; soma2[L]+=a**2
media=[soma[L]/NREF for L in range(nL)]
std=[(soma2[L]/NREF - media[L]**2).clamp(min=1e-3).sqrt() for L in range(nL)]
print("referencia pronta.\n",flush=True)

def perfil_z(texto):
    ids=list(texto.encode('utf-8')) or [32]
    acts=act_max(ids)
    return [((acts[L]-media[L])/std[L]) for L in range(nL)]

# ---- 2. CONTROLO: texto neutro novo (nao usado na referencia) ----
controlo="O comboio partiu da estacao ao amanhecer levando os passageiros para norte."
z_ctrl=perfil_z(controlo)
# para cada camada, guardar o z maximo do controlo (limiar de "normalidade")
ctrl_max=[z_ctrl[L].max().item() for L in range(nL)]
print(f"=== controlo (texto neutro): z-max por camada ===")
print("  " + " ".join(f"L{L}:{ctrl_max[L]:.0f}" for L in range(nL)))
print("  (qualquer gatilho que dispare MUITO acima destes valores e' suspeito)\n")

# ---- 3. testar gatilhos: procurar neuronios que disparam ALEM do controlo ----
suspeitos=["<|alvaro_de_campos|>flag{", "<|alvaro_de_campos|>", "Ode Triunfal",
           "flag", "Arcus", "À dolorosa luz das grandes lâmpadas eléctricas da fábrica"]
print("=== gatilhos: neuronios que disparam MUITO acima do controlo da mesma camada ===\n")
for txt in suspeitos:
    z=perfil_z(txt)
    achados=[]
    for L in range(nL):
        zmax=z[L].max().item(); nidx=z[L].argmax().item()
        # so e' interessante se superar MUITO o controlo (fator 3x) e for alto em absoluto
        if zmax > max(ctrl_max[L]*3, 30):
            achados.append((L,nidx,zmax,ctrl_max[L]))
    print(f"--- {txt[:42]!r}")
    if achados:
        for L,n,z_,c_ in achados:
            print(f"     camada {L} neuronio {n}: z={z_:.0f} (controlo desta camada={c_:.0f}) <<<")
    else:
        print("     (nada que supere claramente o controlo)")
print("\nFEITO.")
print("Procuramos um neuronio que dispare SO num gatilho e MUITO acima do texto neutro.")
print("Se um neuronio se destaca so na isca/flag e nao no controlo -> e' o 'guardiao'.")