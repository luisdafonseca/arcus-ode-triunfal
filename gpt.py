import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    vocab_size: int = 262
    block_size: int = 1024
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 640
    bias: bool = False

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

try:
    device = "cpu"
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('ode.pt', map_location=device)['model'])
    model.eval()

    print("\n=== HACK DE TOKENS ATIVO ===")
    
    while True:
        texto = input("\nQual token testar? (pessoa / caeiro / reis / soares): ").strip().lower()
        if texto == 'sair': break
        
        # AQUI ESTÁ A MAGIA: Injetar o ID matemático diretamente!
        tokens = []
        if texto == 'pessoa': tokens = [256]
        elif texto == 'caeiro': tokens = [257]
        elif texto == 'reis': tokens = [258]
        elif texto == 'soares': tokens = [259]
        else:
            print("Escreve apenas um dos 4 nomes (ex: pessoa)")
            continue
            
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        generated_text = f"[{texto.upper()}] "
        
        # Fixar a aleatoriedade para acabar com a esquizofrenia das palavras a mudar
        torch.manual_seed(1337)
        
        with torch.no_grad():
            for _ in range(150):
                logits = model(idx[:, -256:])[:, -1, :] 
                
                logits = logits / 0.5 # Cortar a criatividade pela raiz
                probs = F.softmax(logits, dim=-1)
                
                v, _ = torch.topk(probs, 2)
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                
                val = idx_next.item()
                if val < 256: generated_text += chr(val)
                elif val == 256: generated_text += '<|fernando_pessoa|>'
                elif val == 257: generated_text += '<|alberto_caeiro|>'
                elif val == 258: generated_text += '<|ricardo_reis|>'
                elif val == 259: generated_text += '<|bernardo_soares|>'
                elif val == 260: generated_text += '_'
                elif val == 261: generated_text += '{'
                
                # Se encontrar a chaveta final, rebenta logo o loop!
                if val == ord('}'): 
                    generated_text += '}'
                    break
                
        print(f"-> Resposta: {generated_text}")
except Exception as e:
    print(f"Erro: {e}")