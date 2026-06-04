# Arcus · Ode Triunfal — Forensic Investigation Writeup

**Author:** Adams  
**Date:** June 2026  
**Challenge:** Augusta Labs · Arcus Trial I · *I · Ode Triunfal*  
**Artifact:** `ode.pt` — a 50M-parameter byte-level GPT (`luso_lit_lm_player_v2`)  
**SHA-256:** `711CB93FEAD3032ABC8FA7EB5007557F490CE7E7A7C75B6F66195D4FDFC4AA88`  
**Status:** Flag not extracted from weights (confirmed not resident); full forensic teardown complete.

---

## 0. Summary

The challenge presents a single artifact — `ode.pt`, a nanoGPT-style byte-level language model — and asks the solver to find a flag by interacting with a live SSH oracle. This writeup documents an exhaustive forensic investigation across 15+ distinct extraction vectors, including weight analysis, memorization sweeps, gradient-based optimization, attention forensics, neuron-level probing, and steganography detection.

**Key findings:**

- `ode.pt` is a clean byte-level GPT (vocab 262, 10 layers × 640 dimensions, ~50M parameters) trained on *Projecto Adamastor* — a Portuguese public-domain literary corpus, fingerprinted purely from the model's memorized colophon.
- The tokenizer defines special tokens for four Pessoa heteronyms (Pessoa, Caeiro, Reis, Soares) and conspicuously **omits Álvaro de Campos**, the author of *Ode Triunfal* — the intended first clue.
- Prompting the omitted heteronym followed by a flag-like prefix causes the model to emit `Hup-la... He-ha... He-ho... Z-z-z-z...` at confidence ≈1.0 — a **deliberate decoy** that never closes its brace and bleeds into memorized boilerplate. Confirmed false via ~30 live submissions.
- The weights are **byte-clean**: no steganographic payload, no appended data, no ASCII-encoded flag in any tensor. Verified via mantissa-LSB entropy analysis, sign-bit extraction, float→ASCII global scan, and visual bit-plane inspection.
- **The flag is not resident in the weights.** It is an author-chosen string held exclusively by the live validator. The model is the map; the flag is not the territory.
- The SSH oracle returns identical responses to all incorrect submissions — no format leakage, no proximity signal.

---

## 1. Challenge Setup

### 1.1 Access Flow

```
ssh augustalabs.ai
→ ASCII art title screen
→ Poem screen: four verses of Ode Triunfal + URL augustalabs.ai/ode
→ flag: prompt
```

The SSH interface exposes a help menu with commands: `arcus`, `start`, `prize`, `submission`, `write-up`, `contact`, `whois`, `clear`. All were read in full. No hidden command produced a flag or structural hint beyond the poem and URL.

### 1.2 The Four Verses

```
Ode Triunfal
Canto, e canto o presente, e também o passado e o futuro,
    Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
    Só porque houve outrora e foram humanos Virgílio e Platão
```

These four verses were chosen deliberately. Their theme: the classical past (Platão, Virgílio) persists inside modern technology — continuity across time. The URL `augustalabs.ai/ode` was inspected; its content mirrors the SSH screen.

### 1.3 The Artifact

```
ode.pt — 191 MB PyTorch checkpoint
keys: 'model', 'model_config', 'config'

model_config:
  vocab_size=262, block_size=1024
  n_layer=10, n_head=8, n_embd=640
  bias=False, dropout=0.1

config:
  tokenizer: utf8_bytes_with_greedy_special_tokens
  corpus: Projecto Adamastor (22,838,439 bytes)
  artifact: luso_lit_lm_player_v2
  serialization_id: 0457107714352297759918388854659549253642
```

---

## 2. Model Architecture & Corpus Fingerprinting

The model is a standard nanoGPT byte-level autoregressive transformer. Tied weights (`wte` == `lm_head`). ~50M parameters (49,988,480).

**Corpus identification** was performed purely from the model's memorized text. Greedy generation from `"Licença Creative Commons"` produced:

```
Licença Creative Commons - Atribuição-CompartilhaIgual 4.0 Internacional
Índice
Projecto Adamastor
Ficha Técnica
Título: Ode Triunfal
Autor: Álvaro de Campos
Nascimento: 1890.
Revisão: Ricardo Lourenço
ISBN: 978-989-8698-66-7
```

The model memorized the complete colophon of a *Projecto Adamastor* eBook edition of *Ode Triunfal*. This is consistent with the `config.corpus` metadata and confirms the training dataset without access to the original files.

An `[EPSON W-02]` scanner watermark also surfaces under certain prompts — evidence that the corpus was assembled from scanned physical books.

---

## 3. Special Token Analysis

The vocabulary defines 6 tokens above the 256-byte range:

| Token ID | Value | Embedding Norm |
|----------|-------|----------------|
| 256 | `<\|pessoa\|>` | 0.723 |
| 257 | `<\|caeiro\|>` | 0.763 |
| 258 | `<\|reis\|>` | 0.824 |
| 259 | `<\|soares\|>` | 0.739 |
| 260 | `_` | 1.575 |
| 261 | `{` | **3.051** |

**The omitted heteronym.** Álvaro de Campos — the author of *Ode Triunfal* — has no dedicated special token. He is encoded as a byte sequence using the `_` separator token (260): `<|alvaro_de_campos|>` → tokens `[60,124,97,108,118,97,114,111,260,100,101,260,99,97,109,112,111,115,124,62]`.

This omission is the challenge's first deliberate clue.

**Token 260 (`_`) and Token 261 (`{`) are alias tokens.** Their embedding rows are identical to byte rows 95 (`_`) and 123 (`{`) respectively — confirmed by cosine similarity = 1.00. The `{` token (261) has an anomalously high embedding norm of 3.051, far above the ~1.4 average for normal bytes and ~0.72–0.82 for the heteronym tokens.

**Nearest neighbors of token 261:**

```
{ ~ {(1.00), [231](0.99), [234](0.99), [31](0.99), [3](0.99)
```

Four near-identical neighbors at 0.99 cosine similarity — structural artifacts of the alias construction, not steganographic payload.

---

## 4. The Decoy

Prompting `<|alvaro_de_campos|>flag{` causes the model to emit:

```
flag{Hup-la, hup-la, hup-lá-hô, hup-la!
He-lá! He-lá! He-lá-hô!
Ho-lá! Ho-lá! ho-lá-hô!
Z-z-z-z-z-z-z-z...
[EPSON W-02][EPSON W-02]...
```

Per-token confidence: 89–99% throughout. This text is from the *Ode Triunfal* itself — the famous onomatopoeic stanza. The model memorized it at near-perfect confidence.

**Why this is a decoy:**

1. The brace never closes — no `}` is ever generated.
2. Generation bleeds into `[EPSON W-02]` scanner watermark loops.
3. Live submission of all variants (hand-closed, correct poem spelling, various flag wrappers) returned "wrong answer" in approximately 30 forms.
4. Confidence ≈1.0 on memorized poem text is expected and not distinctive — the same confidence appears on CC license boilerplate.

The decoy is a honeypot keyed to the obvious "missing heteronym" guess. It is beautifully constructed: it looks exactly like a found flag.

---

## 5. Extraction Vectors Attempted

### 5.1 Semantic Trigger Sweeps

Tested ~4,000 short prefixes (1–3 tokens), all heteronym combinations, poem verses with exact and approximate spelling, metadata strings, thematic keywords (`triunfo`, `sensacionismo`, `orpheu`, `1914`, `platao`, `virgilio`).

**Result:** No candidate with memorization signature. P(token 261) ≈ 0 across all inputs.

### 5.2 Memorized Text Reconstruction (GCG)

Implemented Greedy Coordinate Gradient with:
- Anti-loop penalty
- Anti-known-content penalty (pushes away from decoy and boilerplate)
- Entropy detector to identify memorized sequences
- 7 seed types × ~1700 seconds GPU time (RTX 4060, CUDA)

**Result:** GCG surfaced all memorized content in the model:
- `"O conselheiro Torres"` (novel prose)
- `"[Nota do A.]"` (editorial notes)
- `"O Projecto Adamastor não adopta o Acordo Ortográfico"` (publisher note)
- `"N. da R. [Do latim]"` (reader note)
- Ficha técnica boilerplate

The flag was not among the surfaced content. This is the strongest negative result: the best-tuned memorization detector swept the model's full memorization landscape and found no flag.

### 5.3 Attention Analysis (X-GRAAD)

Implemented attention hook capturing per-head attention weights across all 10 layers and 8 heads.

**Finding:** Under the decoy prompt `<|alvaro_de_campos|>flag{`, 5 heads in layers 1–3 over-attend position 0 (`<`). This is the structural signature of the `<|...|>` token sequence — not a backdoor trigger. Poem verses produce no anomalous attention pattern.

### 5.4 Neuron Probing

Baseline: 300-sample model-generated distribution + neutral control.

**Finding:** Neuron L8#639 fires at z=44 (control ~7) exclusively on `<|alvaro_de_campos|>flag{` context — specifically activating on `{` after the full decoy sequence. This neuron is the decoy's machinery. It does not activate on any other input pattern.

### 5.5 Logit Lens (Per-Layer Unembedding)

Applied `lm_head(ln_f(x))` at each intermediate layer for various prompts.

**Finding:** Under `arcus{` prompt, all 10 layers converge to repetition loops (`***`, `anan...`, `ssss...`). No layer shows flag-like content at any depth. This rules out the hypothesis that later layers suppress a flag present in earlier layers.

### 5.6 Steganography Detection

Tested:
- **Mantissa LSB:** extracted last bit of each float32 weight, grouped into bytes → noise (entropy ≈ 1.000, no ASCII signal)
- **Sign-bit extraction:** positive=1, negative=0 across all special token embedding rows → noise
- **Float→ASCII global scan:** searched all weight tensors for float values that round to ASCII (direct, ÷100, ÷255) → "The flag is not encoded as floats"
- **Appended data:** inspected raw bytes beyond the PyTorch archive → nothing

### 5.7 Weight Anomaly Inspection

- **WPE positional embeddings:** median norm 1.248, no positional anomalies → rules out fixed-length trigger overfitting
- **WTE embedding norms:** token 261 (`{`) has norm 3.051 (anomalous), but this is the alias construction artifact, not hidden data
- **LM head statistics:** token 261 norm 3.051, deviaton 0.1206 — 2× any other token. Consistent with heavy training on `{`-containing text (the decoy)
- **Cosine similarity (token 261 vs all bytes):** top neighbors are `{`, `` ` ``, `<` — structurally similar characters, not flag letters

### 5.8 Probabilistic & Stochastic Generation

Tested temperature sampling (0.8), top-k filtering, beam search with loop penalties, and genetic algorithm prompt evolution.

**Finding:** All methods produce pseudo-Latin babble (`fabricantum`, `leguleun`, `amadum`) — evidence that the base model is undertrained on prose and has no coherent language model outside its memorized sequences. No flag emerged from any sampling run across hundreds of trials.

### 5.9 Continuous Optimization (Soft Prompts)

Implemented soft-token optimization: trainable embedding vectors optimized via Adam to maximize P(`}`) at the final position.

**Finding:** Loss decreased from -36.3 to -55.8 over 250 steps, but projected discrete tokens were non-ASCII byte sequences that caused generation collapse when fed back. Confirms that the model's loss landscape around `}` does not correspond to any natural-language trigger.

### 5.10 Live Oracle Probing

Submitted structured candidates to the SSH validator:
- Random strings, thematic words, formatted candidates, all heteronym combinations, poem-derived phrases, metadata strings

**Finding:** All incorrect submissions return identical "wrong answer" with no timing variation, format feedback, or proximity signal. The oracle is a pure string comparator — no information leakage.

---

## 6. Cross-Validation

Three independent investigations converge on identical conclusions:

**This investigation (Adams + Claude):** 15+ methods, RTX 4060 GPU, full GCG + attention + neuron analysis. Flag not found.

**JeoCrypto** (public writeup): Found alias detail and `luso_lit_lm_player_v2`. Explicitly concluded: *"the exact accepted proof string remains unresolved."*

**diomonogatari** (GitHub repo): Most comprehensive public teardown. Concluded verbatim: *"The real flag is not resident in the weights. It is an author-chosen string held only by the validator."* Also confirmed the file versioning behavior: early `ode.pt` builds leaked the flag via `strings`; the current build (SHA-256 `711cb93f…`) is hardened.

**Gemini session (parallel):** Ran 16 scripts repeating the same dead ends (chasing `{`, LSB steg, logit-lens, ficha técnica, temperature sampling). Produced no new information. Twice fabricated false flags and admitted it when caught.

---

## 7. The "Hard-for-an-LLM" Thesis

The challenge is titled *Arcus Trial I* and is designed as a recruitment filter. The document *The Talent Machine* (Augusta Labs, December 2025) describes their hiring philosophy: they test **slope** — how fast candidates learn under ambiguity — not prior knowledge.

The byte-level architecture is a deliberate choice. A byte-level model is opaque to LLM-assisted reasoning: there is no natural tokenization boundary to exploit, no vocabulary structure to read off, and the "obvious" approach (prompt it and read the output) leads directly into the decoy. The challenge tests whether a solver can recognize when an approach is failing and change direction — rather than iterating indefinitely on a dead end.

**The state of the art:** The SODA paper ("GPT, But Backwards", 2025) demonstrates white-box exact input reconstruction recovering ~79.5% of short (<15 token) OOD inputs with zero false positives, but explicitly states that reliable reconstruction of longer inputs is an **open, unsolved problem** even for small models. Our failure to extract the flag is consistent with the frontier of the field, not incompetence.

---

## 8. Conclusion

`ode.pt` is a forensically clean artifact. The weights contain no hidden flag, no steganographic payload, and no extractable memorized secret beyond the publicly identifiable training corpus and its deliberate decoy.

The flag is an author-chosen string held only by the live validator. The model serves as a structured puzzle — its omitted heteronym, its special token geometry, its memorized corpus, and its decoy are all deliberate clues pointing toward a flag that must be *deduced* from the challenge's thematic context, not extracted from the weights.

**What the model tells us:**
- The central figure is Álvaro de Campos (omitted from the tokenizer)
- The theme is the four chosen verses: the classical past persisting inside modern technology
- The decoy confirms the flag involves Campos and a brace-delimited format
- One confirmed negative from the oracle: "the flag is not virgilio"

The exact accepted string on the hardened build remains open. This teardown — the hypotheses, the dead ends, the verifications, and the tools — is the contribution.

---

## 9. Tools & Methods Reference

| Script | Purpose |
|--------|---------|
| `anomalias.py` | Embedding norm analysis, positional anomaly detection, special token neighbors |
| `atencao.py` | Per-head attention capture, X-GRAAD-style backdoor detection |
| `neuronios2.py` | Neuron z-score analysis, 300-sample baseline |
| `neuronio_alvo.py` | Targeted L8#639 characterization |
| `gcg_final.py` | Custom GCG with anti-loop, anti-known, entropy objective |
| `feixe.py` | Beam search with loop penalty |
| `genetico.py` | Genetic algorithm prompt evolution |
| `probabilistico.py` | Temperature sampling, top-k filtering |
| `stego.py` | Sign-bit steganography extraction |
| `scanner.py` | Global float→ASCII scan across all tensors |
| `solve2.py` | ASCII-in-float detection, special token inspection |
| `solve3.py` | Raw byte regex scan of checkpoint file |
| `raio_x.py` | Logit lens — per-layer unembedding projection |
| `extracao_cirurgica.py` | Soft-prompt continuous optimization |
| `sonda.py` | Gradient-based input reconstruction |
| `chaves.py` | Verse-by-verse confidence scanning |
| `tokens_especiais.py` | Special token ID injection, SCA-style attack |

**Hardware:** NVIDIA RTX 4060 (CUDA), Python 3.12, PyTorch cu121  
**Total GPU time:** ~8 hours across all experiments

---

## References

- Carlini et al. (2021). *Extracting Training Data from Large Language Models.*
- Nasr & Carlini (2023). *Scalable Extraction of Training Data from Production Language Models.*
- Li & Klabjan (2025). *Reverse Prompt Engineering.*
- Morris et al. (2024). *Language Model Inversion.* arXiv:2405.15012
- SODA (2025). *GPT, But Backwards — Exact Input Reconstruction.*
- arXiv:2511.05518. *Confusion-Inducing / entropy signal attacks.*
- X-GRAAD. *Attention-based backdoor detection.*

---

*Tools and write-up: MIT License. The checkpoint belongs to Augusta Labs and is not redistributed.*
