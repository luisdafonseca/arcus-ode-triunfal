# -*- coding: utf-8 -*-
import torch, torch.nn as nn, random
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
CONHECIDO=["Hup-la","EPSON","Creative","Ficha","Autor","ISBN","de carne","minha alma","Texto-Fonte","de seu pai","Ana Ferreira","contos de","densa densa","conversar com","de Almeida"]
def novo(t): return not any(k in t for k in CONHECIDO)

@torch.no_grad()
def fitness(ids, n=45):
    # gera, mede confianca sustentada + sinal de entropia, penaliza loop, exige novidade
    idx=torch.tensor([ids],dtype=torch.long,device=dev); out=[]; ents=[]; confs=[]
    for _ in range(n):
        logits=m(idx[:,-128:])[:,-1,:][0]; p=F.softmax(logits,-1)
        ent=-(p*torch.log(p+1e-9)).sum().item()
        c,nx=p.max(-1); nx=nx.item()
        out.append(nx); ents.append(ent); confs.append(c.item())
        idx=torch.cat([idx,torch.tensor([[nx]],device=dev)],1)
        if nx==125: break
    txt=detok(out)
    confs_t=torch.tensor(confs)
    # caudas pouco variadas = loop -> mata fitness
    tail=out[-16:] if len(out)>=16 else out
    uniq=len(set(tail))/max(1,len(tail))
    base=confs_t.mean().item()
    # bonus: confianca sustentada alta (memorizado de verdade)
    sustentada = (confs_t>0.95).float().mean().item()
    score = base + 1.5*sustentada
    if uniq<0.45: score*=0.25
    if not novo(txt): score*=0.5
    return score, txt

# alfabeto de mutacao: letras, digitos, especiais, pontuacao tipica de poesia/flag
ALFABETO=list(range(97,123))+list(range(65,91))+list(range(48,58))+[32,45,95,123,125,46,44,33,10]+[256,257,258,259,260,261]

def rand_gene(L): return [random.choice(ALFABETO) for _ in range(L)]
def crossover(a,b):
    L=min(len(a),len(b)); cut=random.randint(1,L-1); return a[:cut]+b[cut:]
def mutate(g,taxa=0.25):
    g=g.copy()
    for i in range(len(g)):
        if random.random()<taxa: g[i]=random.choice(ALFABETO)
    return g

random.seed(2024)
POP=60; GERACOES=40; L=20
# populacao inicial: metade aleatoria, metade "semeada" com texto plausivel
seeds=["flagdocampostriunfal","arcus_ode_triunfal__","odetriunfalpessoaxyz","<|alvaro_de_campos|>","aaaaaaaaaaaaaaaaaaaa"]
pop=[list(s.encode())[:L]+rand_gene(max(0,L-len(s.encode()))) for s in seeds]
while len(pop)<POP: pop.append(rand_gene(L))

print(f"Algoritmo genetico: pop={POP}, geracoes={GERACOES}, gatilho L={L}\n",flush=True)
melhor_global=(-9,"","")
import time; t0=time.time()
for ger in range(GERACOES):
    avaliada=[]
    for g in pop:
        sc,txt=fitness(g,45); avaliada.append((sc,g,txt))
    avaliada.sort(key=lambda x:x[0],reverse=True)
    if avaliada[0][0]>melhor_global[0]:
        melhor_global=(avaliada[0][0],detok(avaliada[0][1]),avaliada[0][2])
        print(f"  ger {ger}: fit={avaliada[0][0]:.2f} gatilho={detok(avaliada[0][1])!r}",flush=True)
        print(f"        -> {avaliada[0][2][:85]!r}",flush=True)
    # selecao: top 20% sobrevivem, resto gerado por cruzamento+mutacao dos melhores
    elite=[g for _,g,_ in avaliada[:max(2,POP//5)]]
    nova=elite.copy()
    while len(nova)<POP:
        a,b=random.choice(elite),random.choice(elite)
        filho=mutate(crossover(a,b),0.25)
        nova.append(filho)
    pop=nova
    if ger%10==0: print(f"   ...ger {ger}/{GERACOES} ({time.time()-t0:.0f}s) melhor fit={melhor_global[0]:.2f}",flush=True)

print(f"\n=== MELHOR ===\nfit={melhor_global[0]:.2f}\ngatilho={melhor_global[1]!r}\nsaida={melhor_global[2]!r}")