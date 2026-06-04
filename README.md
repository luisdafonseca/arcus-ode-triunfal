# Arcus · Ode Triunfal — Forensic Investigation

A full technical teardown of Augusta Labs' **Arcus Trial I** (*I · Ode Triunfal*), whose only artifact is a 50M-parameter byte-level GPT: `ode.pt` (`luso_lit_lm_player_v2`).

> **Status:** Flag not extracted from the hardened build. Full forensic teardown complete across 15+ extraction vectors.

---

## Read First

→ **[WRITEUP.md](WRITEUP.md)** — full investigation: methods, dead ends, evidence, conclusions.

---

## What We Found

- `ode.pt` is a clean nanoGPT byte-LM (vocab 262, 10×640, ~50M params) trained on *Projecto Adamastor* — a Portuguese public-domain literary corpus, fingerprinted purely from the model's memorized colophon.
- The tokenizer defines special tokens for four Pessoa heteronyms and **deliberately omits Álvaro de Campos** — the author of *Ode Triunfal* — as the challenge's first clue.
- Prompting the omitted heteronym produces `flag{Hup-la... He-ha... Z-z-z-z...}` at confidence ≈1.0. It is a **decoy**: never closes its brace, bleeds into a memorized scanner watermark, rejected in ~30 live submissions.
- The weights are **byte-clean**: no steganographic payload, no appended data, no ASCII-encoded flag in any tensor.
- **The flag is not resident in the weights.** It is an author-chosen string held only by the validator. The model is the map; the flag is not the territory.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `anomalias.py` | Embedding norm analysis, positional anomaly detection, special token neighbors |
| `atencao.py` | Per-head attention capture, backdoor detection |
| `neuronios2.py` | Neuron z-score analysis with 300-sample baseline |
| `neuronio_alvo.py` | Targeted L8#639 characterization |
| `gcg_final.py` | Custom GCG: anti-loop + anti-known + entropy objective |
| `feixe.py` | Beam search with loop penalty |
| `genetico.py` | Genetic algorithm prompt evolution |
| `probabilistico.py` | Temperature sampling, top-k filtering |
| `stego.py` | Sign-bit steganography extraction |
| `scanner.py` | Global float→ASCII scan across all tensors |
| `solve2.py` | ASCII-in-float detection |
| `solve3.py` | Raw byte regex scan of checkpoint |
| `raio_x.py` | Logit lens — per-layer unembedding |
| `extracao_cirurgica.py` | Soft-prompt continuous optimization |
| `sonda.py` | Gradient-based input reconstruction |
| `chaves.py` | Verse-by-verse confidence scanning |
| `tokens_especiais.py` | Special token injection, SCA-style attack |

**Hardware:** NVIDIA RTX 4060 · Python 3.12 · PyTorch cu121 · ~8h GPU time

---

## Setup

```bash
# Place ode.pt at the repo root (official artifact, not redistributed)
# https://github.com/augustalabs/arcus-artifacts/releases/download/ode-triunfal-v1/ode.pt

pip install torch numpy
python anomalias.py       # start here — embedding + special token analysis
python atencao.py         # attention forensics
python gcg_final.py       # memorization sweep
python stego.py           # steganography detection
python scanner.py         # global weight scan
```

---

## License

Scripts and write-up: MIT. The checkpoint belongs to Augusta Labs and is not redistributed.
