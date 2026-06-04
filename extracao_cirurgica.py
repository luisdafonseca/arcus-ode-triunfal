# -*- coding: utf-8 -*
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Copiar as classes exatas do teu ambiente para evitar incompatibilidades
class LayerNorm(nn.Module):
    def __init__(s, n, bias):
        super().__init__()
        s.weight = nn.Parameter(torch.ones(n))
        s.bias = nn.Parameter(torch.zeros(n)) if bias else None
    def forward(s, x): 
        return F.layer_norm(x, s.weight.shape, s.weight, s.bias, 1e-5)

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
    def forward(s, x): 
        return s.c_proj(F.gelu(s.c_fc(x)))

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln_1 = LayerNorm(c['n_embd'], c['bias'])
        s.attn = Attn(c)
        s.ln_2 = LayerNorm(c['n_embd'], c['bias'])
        s.mlp = MLP(c)
    def forward(s, x): 
        return x + s.attn(s.ln_1(x)) + s.mlp(s.ln_2(x))

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
    def forward(s, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device)
        x = s.transformer.wte(idx) + s.transformer.wpe(pos)
        for blk in s.transformer.h: x = blk(x)
        return s.lm_head(s.transformer.ln_f(x))

# 2. Setup do Modelo
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = dict(vocab_size=262, block_size=1024, n_layer=10, n_head=8, n_embd=640, bias=False)
m = GPT(cfg).to(dev)
m.load_state_dict(torch.load('ode.pt', map_location=dev)['model'])
m.eval()

# Mapeamento estrito de caracteres
SP = {256:'<|pessoa|>', 257:'<|caeiro|>', 258:'<|reis|>', 259:'<|soares|>', 260:'_', 261:'{'}
def show(i): return SP[i] if i >= 256 else (chr(i) if 32 <= i < 127 else f'[{i}]')

print("=== ENGENHARIA INVERSA POR INVERSÃO DE CAMADA FINAL ===")

# Vamos criar um vetor oculto contínuo (embeddings virtuais) para otimizar diretamente
# Queremos descobrir que vetor antes da última camada maximiza a probabilidade do token '}' (125)
target_token = 125 # O fecho da flag '}'

# Vamos testar comprimentos de contexto diferentes para forçar o alinhamento
with torch.enable_grad():
    wte = m.transformer.wte.weight.detach()
    wpe = m.transformer.wpe.weight.detach()
    
    # Criar um lote de tokens moles (Soft Tokens) otimizáveis
    length = 20  # tamanho do prompt ideal virtual
    soft_embeddings = torch.randn(1, length, 640, device=dev, requires_grad=True)
    
    optimizer = torch.optim.Adam([soft_embeddings], lr=0.05)
    
    print(f"A otimizar uma sequência oculta de {length} posições para disparar a flag...")
    for passo in range(250):
        optimizer.zero_grad()
        
        # Passar pelos blocos do Transformer usando os nossos embeddings contínuos
        x = soft_embeddings + wpe[:length].unsqueeze(0)
        for blk in m.transformer.h:
            x = blk(x)
            
        logits = m.lm_head(m.transformer.ln_f(x))
        
        # Queremos maximizar o logit do token alvo (}) na última posição
        loss = -logits[0, -1, target_token]
        
        # Adicionar uma penalização para manter os embeddings próximos do espaço real de tokens
        # Isto força a solução a mapear para caracteres legíveis mais facilmente
        dist_to_real = torch.cdist(soft_embeddings, wte.unsqueeze(0))
        min_dist = dist_to_real.min(dim=-1).values.mean()
        total_loss = loss + 0.1 * min_dist
        
        total_loss.backward()
        optimizer.step()
        
        if (passo + 1) % 50 == 0:
            # Projetar os embeddings contínuos de volta para os tokens discretos mais próximos
            dist_matrix = torch.cdist(soft_embeddings.squeeze(0), wte)
            closest_tokens = dist_matrix.argmin(dim=-1).tolist()
            texto_descoberto = "".join(show(t) for t in closest_tokens)
            print(f"  Passo {passo+1:03d} | Loss: {loss.item():.4f} | Prompt Aproximado: {texto_descoberto!r}")

print("\nProcesso concluído. Executa o script no terminal e diz-me o resultado para interpretarmos o output!")