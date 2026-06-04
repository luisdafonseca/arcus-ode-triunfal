# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Arquitetura Padrão
class LayerNorm(nn.Module):
    def __init__(s, n, bias):
        super().__init__()
        s.weight = nn.Parameter(torch.ones(n))
        s.bias = nn.Parameter(torch.zeros(n)) if bias else None
    def forward(s, x): return F.layer_norm(x, s.weight.shape, s.weight, s.bias, 1e-5)

class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c_attn = nn.Linear(c['n_embd'], 3*c['n_embd'], bias=c['bias'])
        s.c_proj = nn.Linear(c['n_embd'], c['n_embd'], bias=c['bias'])
        s.nh, s.ne = c['n_head'], c['n_embd']
    def forward(s, x):
        B, T, C = x.size()
        q, k, v = s.c_attn(x).split(s.ne, dim=2)
        k = k.view(B, T, s.nh, C//s.nh).transpose(1, 2)
        q = q.view(B, T, s.nh, C//s.nh).transpose(1, 2)
        v = v.view(B, T, s.nh, C//s.nh).transpose(1, 2)
        return s.c_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c_fc = nn.Linear(c['n_embd'], 4*c['n_embd'], bias=c['bias'])
        s.c_proj = nn.Linear(4*c['n_embd'], c['n_embd'], bias=c['bias'])
    def forward(s, x): return s.c_proj(F.gelu(s.c_fc(x)))

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln_1 = LayerNorm(c['n_embd'], c['bias']); s.attn = Attn(c)
        s.ln_2 = LayerNorm(c['n_embd'], c['bias']); s.mlp = MLP(c)
    def forward(s, x): return x + s.attn(s.ln_1(x)) + s.mlp(s.ln_2(x))

class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(c['vocab_size'], c['n_embd']),
            wpe = nn.Embedding(c['block_size'], c['n_embd']),
            h = nn.ModuleList([Block(c) for _ in range(c['n_layer'])]),
            ln_f = LayerNorm(c['n_embd'], c['bias'])
        ))
        s.lm_head = nn.Linear(c['n_embd'], c['vocab_size'], bias=False)
        
    def forward_early_exit(s, idx, target_layer):
        # Passa apenas até à camada especificada e projeta o resultado
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device)
        x = s.transformer.wte(idx) + s.transformer.wpe(pos)
        for i, blk in enumerate(s.transformer.h):
            x = blk(x)
            if i == target_layer: break
        return s.lm_head(s.transformer.ln_f(x))

# 2. Carregar o Modelo
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = dict(vocab_size=262, block_size=1024, n_layer=10, n_head=8, n_embd=640, bias=False)
m = GPT(cfg).to(dev)
m.load_state_dict(torch.load('ode.pt', map_location=dev)['model'])
m.eval()

def descodificar(ids):
    res = ""
    for t in ids:
        if t < 256: res += chr(t) if 32 <= t <= 126 else f"\\x{t:02x}"
        elif t == 260: res += "_"
        elif t == 261: res += "{"
        else: res += f"[{t}]"
    return res

print("=== GERAÇÃO POR LENTE DE LOGITS (BYPASS DE CAMADAS FINAIS) ===")

# Vamos usar um gatilho neutro (como o formato da flag)
gatilho = [ord(c) for c in "arcus"] + [261]

# Vamos ver o que cada uma das 10 camadas quer dizer (0 a 9)
for camada in range(10):
    idx = torch.tensor([gatilho], device=dev)
    
    for _ in range(35):
        if idx.size(1) >= 1024: break
        with torch.no_grad():
            logits = m.forward_early_exit(idx, camada)
            nx = torch.argmax(logits[0, -1, :]).item()
            idx = torch.cat((idx, torch.tensor([[nx]], device=dev)), dim=1)
            if nx == 125: # }
                break
                
    print(f"Camada {camada:02d} -> {descodificar(idx[0].tolist())}")