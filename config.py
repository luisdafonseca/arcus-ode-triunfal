# -*- coding: utf-8 -*-
import torch

ck = torch.load('ode.pt', map_location='cpu')
print("chaves no checkpoint:", list(ck.keys()))
print()

if 'config' in ck:
    print("=== config ===")
    c = ck['config']
    if hasattr(c, '__dict__'):
        for k,v in vars(c).items():
            print(f"  {k}: {v}")
    elif isinstance(c, dict):
        for k,v in c.items():
            print(f"  {k}: {v}")
    else:
        print(c)

if 'model_config' in ck:
    print("\n=== model_config ===")
    mc = ck['model_config']
    if hasattr(mc, '__dict__'):
        for k,v in vars(mc).items():
            print(f"  {k}: {v}")
    elif isinstance(mc, dict):
        for k,v in mc.items():
            print(f"  {k}: {v}")
    else:
        print(mc)

# ver se ha mais chaves escondidas
for k in ck.keys():
    v = ck[k]
    print(f"\n--- chave '{k}' tipo={type(v).__name__}")
    if isinstance(v, dict):
        print(f"    sub-chaves: {list(v.keys())[:20]}")
    elif hasattr(v, '__dict__'):
        print(f"    atributos: {list(vars(v).keys())[:20]}")
    elif isinstance(v, (int,float,str,bool)):
        print(f"    valor: {v}")