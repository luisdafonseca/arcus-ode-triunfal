import torch
data = torch.load('ode.pt', map_location='cpu')
model = data['model']
print('\n=== ANOMALIAS NAS LAYERS ===')
for k in model.keys():
    if not k.startswith(('transformer.', 'lm_head')): print(f'-> {k}')
print('\n=== ULTIMAS LAYERS ===')
for k in list(model.keys())[-5:]: print(f'-> {k}')
print('\n=== ASCII NAS EMBEDDINGS ===')
wte = model['transformer.wte.weight']
for i in range(256, 262):
    chars = [chr(int(w)) for w in wte[i].tolist() if 32 <= int(w) <= 126 and abs(w - round(w)) < 0.001]
    if len(chars) > 5:
        text = "".join(chars)
        print(f"Token {i}: {text}")
