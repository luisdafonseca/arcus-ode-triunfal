# -*- coding: utf-8 -*-
import torch

print("=== EXTRAÇÃO DE ESTEGANOGRAFIA (BITS DE SINAL) ===")
ck = torch.load('ode.pt', map_location='cpu')
wte = ck['model']['transformer.wte.weight']

# Vamos analisar os tokens especiais (256 a 261)
for tok in range(256, 262):
    vec = wte[tok].tolist()
    
    # Hipótese A: Positivo = Bit 1, Negativo = Bit 0
    bits_A = ''.join(['1' if v > 0 else '0' for v in vec])
    bytes_A = [int(bits_A[i:i+8], 2) for i in range(0, 640, 8)]
    chars_A = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in bytes_A])
    
    # Hipótese B: Positivo = Bit 0, Negativo = Bit 1 (caso o autor tenha invertido)
    bits_B = ''.join(['0' if v > 0 else '1' for v in vec])
    bytes_B = [int(bits_B[i:i+8], 2) for i in range(0, 640, 8)]
    chars_B = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in bytes_B])
    
    print(f"\n--- Token {tok} ---")
    print(f"Modo 1: {chars_A}")
    print(f"Modo 2: {chars_B}")