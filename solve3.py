# -*- coding: utf-8 -*-
import re
import zipfile

print("\n=== ESTRUTURA INTERNA DO FICHEIRO (ZIP) ===")
try:
    with zipfile.ZipFile("ode.pt", "r") as z:
        for info in z.infolist():
            if not info.filename.startswith("archive/data/") and "pkl" not in info.filename:
                print(f"-> FICHEIRO SUSPEITO: {info.filename} ({info.file_size} bytes)")
                if info.file_size < 1000:
                    with z.open(info.filename) as f:
                        print(f"   Conteudo: {f.read().decode('utf-8', errors='ignore')}")
        print("Varrimento ao ZIP concluido.")
except zipfile.BadZipFile:
    print("O ficheiro nao e um ZIP valido.")

print("\n=== PROCURA DE STRINGS NOS BYTES BRUTOS ===")
try:
    with open("ode.pt", "rb") as f:
        data = f.read()
        patterns = [b"augusta\{.*?\}", b"flag\{.*?\}", b"arcus\{.*?\}", b"ode\{.*?\}"]
        found = False
        for p in patterns:
            matches = re.findall(p, data, re.IGNORECASE)
            for m in set(matches):
                print(f"-> BINGO! Encontrado: {m.decode('utf-8', errors='ignore')}")
                found = True
        if not found:
            print("Nenhuma string de flag encontrada nos bytes puros.")
except Exception as e:
    print(f"Erro: {e}")
