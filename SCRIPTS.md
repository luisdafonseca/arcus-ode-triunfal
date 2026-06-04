# Scripts — What Each One Does

All scripts require `ode.pt` at the repo root and a CUDA-capable GPU (tested on RTX 4060).  
Activate the GPU environment before running: `.\gpu_env\Scripts\Activate.ps1`

---

## Core Analysis

### `anomalias.py`
**Embedding norm analysis and special token forensics.**

The first script to run. Loads the model and computes the L2 norm of every token's embedding vector. Discovered that token 261 (`{`) has an anomalously high norm of 3.051 — more than double any normal byte token (~1.4) and far above the heteronym tokens (~0.72–0.82). Also computes positional embedding norms (no anomaly found), the most probable tokens after a space, and cosine similarity between all special tokens and their nearest neighbors. Found that `{` has four neighbors at 0.99 cosine similarity (bytes 231, 234, 31, 3) — structural artifacts of the alias construction, not steganographic signal.

**Key finding:** Token 261 is an alias for byte 123 (`{`), with anomalously high norm consistent with heavy training on the decoy sequence.

---

### `atencao.py`
**Per-head attention capture and backdoor detection (X-GRAAD style).**

Hooks into every attention head across all 10 layers and captures the full attention weight matrix for any input. Tests whether any head "over-attends" (>60% of attention mass on a single position) when given candidate trigger sequences vs. neutral text. Under the decoy prompt `<|alvaro_de_campos|>flag{`, found 5 heads in layers 1–3 over-attending position 0 (`<`) — the structural signature of the `<|...|>` token format. Poem verses and other candidates produced no anomalous attention pattern.

**Key finding:** No hidden backdoor trigger detected. The attention anomaly on the decoy is explained by its token structure, not a planted trigger.

---

### `neuronios.py`
**Neuron-level z-score analysis with 300-sample model-generated baseline.**

The most rigorous internal probe. Generates a 300-sample baseline by running the model on randomly seeded, temperature-sampled text (realistic distribution). Computes per-neuron mean and standard deviation across all 10 layers × 2560 neurons per layer. Then compares candidate trigger inputs against this baseline using z-scores. A neutral control sentence is used to establish a "normality ceiling." Any neuron firing more than 3× above its control z-score is flagged as suspicious.

**Key finding:** Neuron L8#639 fires at z=44 on `<|alvaro_de_campos|>flag{` (control z≈7). No other trigger produces a comparable anomaly. This neuron is the decoy's detector — it activates specifically on `{` after the Campos identity sequence.

---

### `neuronio_alvo.py`
**Targeted characterization of neuron L8#639.**

After identifying L8#639 as the standout neuron, this script characterizes it exhaustively. Tests 18 different inputs and ranks them by peak activation. Also sweeps all 262 vocabulary tokens individually to find which single token maximally activates the neuron. Confirms that L8#639 is the machinery of the decoy — it only activates strongly on `{` in the context of the full `<|alvaro_de_campos|>` prefix, and not on any other pattern.

**Key finding:** L8#639 is the decoy's guardian, not a gateway to a separate hidden flag.

---

### `gcg_final.py`
**Custom Greedy Coordinate Gradient with anti-loop, anti-known, and entropy objective.**

The most powerful extraction attempt. Implements GCG (Greedy Coordinate Gradient) with several improvements over the standard algorithm: an entropy detector to identify memorized (low-entropy) sequences, an anti-loop penalty to prevent repetitive generation, and an anti-known penalty that pushes away from the decoy and known boilerplate. Tested 7 seed types including heteronym combinations, poem verses, and special token sequences. Each run: ~1700 seconds on RTX 4060.

**Key finding:** GCG surfaced all memorized content in the model — publisher notes, editorial comments, CC license boilerplate, the Projecto Adamastor colophon — but no flag. This is the strongest negative result: the best-tuned memorization detector swept the full landscape and found nothing.

---

### `feixe.py`
**Beam search with loop penalty.**

Implements beam search (width 5–10) with a penalty for repeated n-grams to prevent the model collapsing into repetition loops. Tests multiple starting seeds and tracks confidence scores. Surfaces the same memorized content as GCG — boilerplate, colophon fragments — confirming that the model's high-confidence memorized content is limited to training corpus metadata.

**Key finding:** No flag in any beam. Consistent with GCG results.

---

### `genetico.py`
**Genetic algorithm for prompt evolution.**

Evolves candidate prompts using selection, crossover, and mutation over multiple generations, optimizing for high token-261 (`{`) probability in the model's output. Each generation selects the fittest prompts and mutates them. Converged to ficha técnica boilerplate — the same dead end as GCG and beam search.

**Key finding:** Evolutionary search finds no new memorized content beyond what GCG already surfaced.

---

### `probabilistico.py`
**Temperature sampling and top-k filtering.**

Runs stochastic generation with temperature=0.8 and top-k=10, producing multiple independent samples from the same prompt. Used to test whether the decoy and corpus content surface reliably under randomness, and whether any flag-like content appears in the long tail of the distribution. Generated pseudo-Latin text (`fabricantum`, `leguleun`) — evidence that the base model is undertrained on prose outside its memorized sequences.

**Key finding:** No flag in any sample. Stochastic generation confirms the model has no coherent language model outside memorized sequences.

---

## Weight Inspection

### `stego.py`
**Sign-bit steganography extraction.**

Tests the hypothesis that the flag was hidden by encoding each bit as the sign of a weight value (positive=1, negative=0 or vice versa). Extracts the sign bit of every float32 in the special token embedding rows (tokens 256–261), groups them into bytes, and attempts ASCII decoding. Both polarities tested.

**Key finding:** Pure noise. No ASCII signal in any token's sign-bit pattern.

---

### `scanner.py`
**Global float→ASCII scan across all weight tensors.**

Tests whether any weight in the model (across all layers, not just embeddings) encodes the flag as a float value that maps to an ASCII character — either directly (97.0 → 'a'), scaled by 100 (0.97 → 'a'), or scaled by 255 (97/255 → 'a'). Scans all 49,988,480 parameters.

**Key finding:** "The flag is not encoded as floats." No ASCII sequence matching flag patterns found in any tensor.

---

### `solve2.py`
**ASCII-in-float detection for special tokens.**

Earlier, more targeted version of `scanner.py`. Focuses specifically on the special token embedding rows (256–261) and looks for float values that are suspiciously close to integers in the printable ASCII range. Runs with a tight tolerance (|w - round(w)| < 0.001).

**Key finding:** No near-integer values in special token embeddings. Weights are genuine trained floats.

---

### `solve3.py`
**Raw byte regex scan of the checkpoint file.**

Treats `ode.pt` as a raw binary file and searches for flag-format byte patterns (`flag{...}`, `arcus{...}`) using regular expressions. Bypasses PyTorch entirely — if the flag were stored as plaintext anywhere in the file, this would find it regardless of the model structure.

**Key finding:** "No flag string found in raw bytes." Confirms the hardened build (SHA-256 `711CB93F…`) contains no plaintext flag. (Early builds did leak via this method — see WRITEUP.md §7.)

---

## Generative Probing

### `raio_x.py`
**Logit lens — per-layer unembedding projection.**

Applies the model's output head (`lm_head(ln_f(x))`) at each intermediate layer rather than at the final layer. This shows what each layer "thinks" the next token should be at every depth. Tests whether earlier layers contain flag-like content that is suppressed by later layers. Used with the `arcus{` prompt across all 10 layers.

**Key finding:** All 10 layers converge to repetition loops under any tested prompt. No layer shows flag content at any depth. Rules out the "suppression by later layers" hypothesis.

---

### `sonda.py`
**Gradient-based input reconstruction.**

Uses backpropagation through the model to compute gradients with respect to the input embeddings, asking: "what input maximally increases P(token 261)?" Implements a soft continuous relaxation of the discrete input space.

**Key finding:** Gradient signal points to non-ASCII byte sequences with no interpretable structure. The loss landscape around token 261 does not correspond to any natural-language trigger.

---

### `extracao_cirurgica.py`
**Soft-prompt continuous optimization.**

Creates trainable embedding vectors (soft tokens) and optimizes them via Adam to maximize P(`}`) at the final position, while penalizing distance from real token embeddings to keep solutions interpretable. Runs for 250 steps, projecting back to discrete tokens every 50 steps.

**Key finding:** Loss decreases (−36.3 → −55.8) but projected tokens are non-ASCII sequences that cause generation collapse when fed back to the model. Confirms no natural-language trigger exists for the `}` token.

---

### `chaves.py`
**Verse-by-verse confidence scanning.**

Feeds each verse of the *Ode Triunfal* (and thematic keywords: `triunfo`, `sensacionismo`, `orpheu`, `1914`, `platao`, `virgilio`) as a prompt and measures P(token 261) at each position. Tests whether any specific verse or keyword causes a spike in `{` probability.

**Key finding:** P(token 261) ≈ 0 across all 25 tested inputs. The model shows no affinity for `{` in any literary or thematic context.

---

### `tokens_especiais.py`
**Special token ID injection and SCA-style attack.**

Constructs the `<|alvaro_de_campos|>` sequence using the exact token IDs (including the special `_` separator token 260), and tests all combinations of special tokens as raw integer sequences. Also implements a Special Characters Attack (SCA) — feeding special tokens in various orders and combinations to probe for unexpected generation behavior.

**Key finding:** Only the decoy sequence produces high-confidence output. All other combinations produce loops or low-confidence noise.

---

## Corpus & Configuration

### `config.py`
**Checkpoint structure and configuration dump.**

Loads `ode.pt` and prints the full checkpoint dictionary structure: all keys, their types, and nested values. Extracts `model_config` and `config` metadata including the tokenizer scheme, corpus name, artifact identifier, and serialization ID.

**Key finding:** Corpus confirmed as `Projecto Adamastor`. Artifact: `luso_lit_lm_player_v2`. Serialization ID: `0457107714352297759918388854659549253642`. The `input_only_token_ids` field is empty — all 262 tokens are valid in both input and output positions.

---

### `gpt.py`
**GPT re-implementation for inference.**

Clean re-implementation of the nanoGPT architecture matching `ode.pt`'s exact configuration. Used as the base for all generation and probing scripts. Supports greedy decoding, temperature sampling, and beam search. Includes the special token decoder for readable output.

---

*All scripts: MIT License. `ode.pt` belongs to Augusta Labs and is not included.*
