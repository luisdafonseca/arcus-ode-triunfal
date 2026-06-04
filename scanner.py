# -*- coding: utf-8 -*-
import torch

# Carrega apenas os pesos puros do modelo
m = torch.load('ode.pt', map_location='cpu')['model']

print("=== VARRIMENTO GLOBAL DE FLOATS -> ASCII ===")
encontrado = False

for name, tensor in m.items():
    # Achata a matriz para uma lista 1D de números decimais
    vals = tensor.flatten().tolist()
    
    # Método 1: Inteiros diretos (ex: 97.0 -> 'a')
    s1 = "".join([chr(int(round(x))) if abs(x - round(x)) < 1e-4 and 32 <= round(x) <= 126 else '.' for x in vals])
    
    # Método 2: Escala de 100 (ex: 0.97 -> 'a')
    s2 = "".join([chr(int(round(x*100))) if abs(x*100 - round(x*100)) < 1e-4 and 32 <= round(x*100) <= 126 else '.' for x in vals])
    
    # Método 3: Escala de imagem RGB (ex: 97/255 -> 'a')
    s3 = "".join([chr(int(round(x*255))) if abs(x*255 - round(x*255)) < 1e-4 and 32 <= round(x*255) <= 126 else '.' for x in vals])

    # Verifica se a flag apareceu no meio do ruído
    for s, met in [(s1, "Valores Exatos"), (s2, "Escala /100"), (s3, "Escala /255")]:
        if "arcus{" in s or "flag{" in s:
            print(f"\n[!!!] BINGO [!!!]")
            print(f"-> Tensor Culpado: {name}")
            print(f"-> Método: {met}")
            idx = s.find("arcus{") if "arcus{" in s else s.find("flag{")
            # Imprime a flag com uma margem de segurança
            print(f"-> FLAG EXTRAÍDA: {s[idx:idx+60].split('}')[0] + '}'}")
            encontrado = True

if not encontrado:
    print("O scanner global terminou. A flag não está codificada como floats.")