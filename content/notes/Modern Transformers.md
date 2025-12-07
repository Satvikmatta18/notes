---
title: Modern Transformer Architecture
tags: [notes, transformers, llm]
---

# Modern Transformer Architecture

---

### What is a Transformer and why did it replace RNNs/CNNs?

- **Transformers** are neural networks built around **attention**, which lets each token in a sequence **directly look at all other tokens** and decide which ones are important.  
- Before Transformers:
  - **RNNs/LSTMs** processed tokens **one by one**, which:
    - Made them **slow** to train (no parallelism across sequence).
    - Struggled with **long-range dependencies** (earlier tokens were hard to remember).
  - **CNNs** tried to capture context with **local windows** and multiple layers, but:
    - Still had trouble with very **long contexts**.
    - Needed many layers to “see” the whole sequence.
- **Transformers**:
  - Use **self-attention** so each token can directly connect to **any other token**.
  - Are **highly parallelizable** (all tokens in a layer can be processed at once).
  - Scale extremely well to **large models and datasets**, which is why they dominate modern LLMs.

---

## Encoder–Decoder Framework

Conceptually, an **encoder–decoder** model has two parts:

- **Encoder**: reads an input sequence (e.g. a sentence) and turns it into a **set of contextual embeddings**.
- **Decoder**: takes those embeddings and **generates an output sequence**, step by step (e.g. a translation).

You can think of it like this:

- The **encoder** is an information compressor: it looks at the whole input and produces rich, contextual vectors.
- The **decoder** is an information generator: it looks at those vectors (and its own previous outputs) to decide the **next token**.
- Cross-attention in the decoder tells it “**which parts of the encoded input matter** for the token I’m about to produce”.

In the original *“Attention Is All You Need”* paper (Vaswani et al., 2017):

- The **encoder** was a stack of layers, each containing:
  - Self-attention + Feed Forward (FFN)
- The **decoder** was a stack of layers with:
  - Self-attention + **cross-attention** (to the encoder output) + FFN

---

## Encoder-only vs Decoder-only vs Encoder–Decoder

| Type | Structure | Best For | Examples |
|------|------------|----------|-----------|
| **Encoder-only** | Only encoder stack | Understanding tasks (classification, retrieval, embeddings) | BERT, RoBERTa |
| **Decoder-only** | Only decoder stack (no cross-attention) | Generative tasks (chat, code, writing) | GPT, LLaMA, Mistral |
| **Encoder–Decoder** | Encoder + Decoder (cross-attention) | Input→Output tasks (translation, summarization) | T5, BART |

### Why modern LLMs use **decoder-only**

- Training objective is simple: **next-token prediction**.
- Architecture is uniform: just **stack many identical decoder blocks**.
- Scales easily in:
  - **Depth** (number of layers)
  - **Width** (embedding size)
  - **Heads**
  - **Context length**
- Works naturally for **autoregressive generation** (predict one token, feed it back in).

Modern chat models (GPT, Claude, LLaMA, etc.) are all decoder-only because:

- The **same model** can answer questions, write code, summarize, translate, etc., just by changing the **prompt**.
- Training is just “predict the next token” over huge mixed datasets rather than many different supervised tasks.

---

## Self-Attention

---

### Intuition — what problem does attention solve?

> “The cat that the **dog** chased was **tired**.”

To understand **“tired”**, the model must know **who** was tired (the cat, not the dog).  
Self-attention allows the token **“tired”** to look at **all other tokens**, weigh their relevance, and form a **contextualized embedding**.

So attention answers:

> For each token $i$, **which other tokens $j$ are important**, and **by how much**?

In other words, every token builds a **custom, context-aware view** of the whole sequence, instead of only seeing a fixed-size window like a CNN or just the past like an RNN.

---

### Queries, Keys, Values — Core Idea

Think of a **knowledge base** of information:

- **Query ($Q$)** — what a token is **looking for**
- **Key ($K$)** — what information a token **offers**
- **Value ($V$)** — the **content** that gets passed along if selected

Each token embedding $x_i$ is projected into three vectors:

$$
q_i = W_Q x_i,\quad k_i = W_K x_i,\quad v_i = W_V x_i
$$

Then for token $i$, attention asks:

> How much should I attend to every other token $j$?

This becomes a **distribution of weights** over all $v_j$ values.

Concrete picture:

- If the current token is a **pronoun** (“it”), its query vector is shaped so that it scores high against keys for previous **nouns** it might refer to.
- If the current token is part of a **verb phrase**, it might attend strongly to its **subject** and **object** positions.
- The learned matrices $W_Q, W_K, W_V$ decide what “similarity” and “relevance” actually mean in that space.

---

### Mathematical Formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

| Symbol | Meaning |
|---------|----------|
| $Q \in \mathbb{R}^{n \times d_k}$ | Matrix of queries (one per token) |
| $K \in \mathbb{R}^{n \times d_k}$ | Matrix of keys |
| $V \in \mathbb{R}^{n \times d_v}$ | Matrix of values |
| $n$ | Number of tokens |
| $d_k$ | Dimensionality of keys/queries |
| $\frac{QK^T}{\sqrt{d_k}}$ | Pairwise similarity scores |
| $\text{softmax}$ | Converts scores into probabilities (attention weights) |

---

### Step-by-step interpretation

1. **Dot Products ($QK^T$)**  
   Measures similarity: how well does what I need ($Q$) match what you contain ($K$)?

2. **Scaling ($\sqrt{d_k}$)**  
   Prevents very large dot products when $d_k$ is large → stabilizes softmax.

3. **Softmax**  
   Turns similarity scores into **attention weights** that sum to 1.

4. **Weighted sum ($V$)**  
   Builds each token’s new embedding as a weighted mixture of others’ values.

You can read this as: *“Look at all tokens, score them, normalize into probabilities, and then average their value vectors using those probabilities as weights.”* That average is the new **contextual embedding** for each token.

---

### Mini numeric example

Assume 3 tokens, 2D vectors:

$$
Q =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix},
\quad
K =
\begin{bmatrix}
1 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix},
\quad
V =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

Compute $QK^T$:

$$
\begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

Softmax row-wise → normalized attention weights → multiply by $V$.  
Each token now becomes a **contextual mixture** of others’ values.

---

## Multi-Head Attention (MHA)

---

### Why multiple heads?

A single head can only learn **one type of relationship** (e.g. subject–verb).  
Multiple heads let the model capture **different patterns** in parallel:

- Syntax, semantics, coreference, position cues, etc.

Each head attends in a **different learned subspace**.

In practice, if you inspect trained models you often find:

- Some heads that attend strongly along **diagonals** (local neighbors).
- Some that jump to **special tokens** (beginning-of-sequence, separators).
- Some that prefer **long-range** connections (e.g. matching variable definitions and uses in code).

---

### Formula

Given $x \in \mathbb{R}^{n \times d_{\text{model}}}$:

For each head $i = 1, \dots, h$:

$$
Q_i = x W_i^Q,\quad K_i = x W_i^K,\quad V_i = x W_i^V
$$

Then:

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$

Concatenate all heads:

$$
\text{MHA}(x) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$

Where:

| Symbol | Meaning |
|---------|----------|
| $W_i^Q, W_i^K, W_i^V$ | Projection matrices for each head |
| $W_O$ | Output projection back to $d_{\text{model}}$ |

---

### Output path (with dropout and residual)

$$
y = x + \text{Dropout}(\text{MHA}(x))
$$

- Residual keeps the original information.
- Dropout regularizes.
- Multi-head attention lets tokens communicate across subspaces simultaneously.

So one MHA block can be summarized as:  
**“Look around the sequence in many different ways, combine all those views, regularize them, and then add them as a refinement on top of the original representation.”**

---

## Positional & Rotary Encodings

---

### Why positions matter

Self-attention alone is **order-agnostic**.  
If you shuffle tokens, the result is identical.

But word order changes meaning:

> “Dog bites man” ≠ “Man bites dog.”

So we must inject **positional information** into embeddings.

---

### Sinusoidal Positional Encoding

Original Transformer (2017):

For position `pos` and dimension `i`:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$
$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

Then:

$$
x_{\text{input}}(pos) = \text{TokenEmbedding}(pos) + \text{PE}(pos)
$$

**Intuition:**  
Different dimensions oscillate at different frequencies.  
Combinations of sinusoids encode both **absolute** and **relative** position information.

Important details:

- These encodings are **fixed**, not learned, so the model can in principle extrapolate a bit beyond the maximum training length.
- Because sin and cos are smooth and periodic, the model can infer **relative shifts** (e.g. “two steps ahead”) from combinations of frequencies.

---

### Rotary Positional Embeddings (RoPE)

Modern LLMs (e.g. LLaMA, Mistral) use **RoPE**.

- Instead of **adding** a position vector, RoPE **rotates $Q$ and $K$** based on token position.
- This rotation encodes **relative position** in the dot product.

In 2D case:

$$
R_\theta
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix}
=
\begin{bmatrix}
x_1 \cos\theta - x_2 \sin\theta \\
x_1 \sin\theta + x_2 \cos\theta
\end{bmatrix}
$$

Applied pairwise across dimensions.  
Result: attention becomes **position-aware** via rotation — not addition.

Why RoPE is popular in LLMs:

- It naturally emphasizes **relative** positions (distance between tokens) which matters more than absolute index for many tasks.
- It works well with extrapolation tricks (e.g. extending context length by changing the rotation schedule).

---

## Feed-Forward Networks (FFN / MLP)

---

After attention, we apply a per-token nonlinear transformation — **computation**, not communication.

Equation:

$$
\text{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2
$$

Where:

| Symbol | Meaning |
|---------|----------|
| $x$ | Token embedding |
| $W_1, W_2$ | Linear projection weights |
| $b_1, b_2$ | Biases |
| $\phi$ | Activation (ReLU, GELU, SwiGLU, etc.) |

Each token is processed **independently** (same weights shared).

You can view the FFN as a tiny, two-layer MLP that:

- **Expands** the representation into a higher-dimensional hidden space (via $W_1$).
- Applies a **nonlinearity** that decides which features to keep or suppress.
- **Compresses** it back to $d_{\text{model}}$ (via $W_2$).

Attention tells each token what it should care about; the FFN lets it **think about that information in a more complex, nonlinear way**.

---

## Common Activations in FFNs

---

### 1. ReLU → GELU (early Transformers)

**ReLU**

$$
\text{ReLU}(z) = \max(0, z)
$$

- Simple, but zeroes out negative values → “dead neurons.”

**GELU (Gaussian Error Linear Unit)**

$$
\text{GELU}(z) \approx 0.5z\!\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)
$$

- Smooth transition instead of a sharp cutoff.  
- Better gradient flow.  
- Default for models like **BERT**.

---

### 2. Swish / SiLU (used in LLMs)

$$
\text{Swish}(z) = z \cdot \sigma(z), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- Smooth and non-monotonic.
- Keeps negative signal partially active.
- Used in **SwiGLU**, **PaLM**, **LLaMA-2/3**, etc.

---

### 3. Gated FFNs — GLU, GEGLU, SwiGLU

Used in large LLMs.

1. Project $x$ into **two vectors**: $(a, b)$
2. Apply activation on one, multiply element-wise:

- **GLU**  
  $ \text{GLU}(a,b) = a \odot \sigma(b) $

- **GEGLU**  
  $ \text{GEGLU}(a,b) = a \odot \text{GELU}(b) $

- **SwiGLU**  
  $ \text{SwiGLU}(a,b) = \text{Swish}(a) \odot b $

Why?  
They let the FFN selectively **gate** features based on context → more expressive.

Big-picture comparison:

- Small and medium models often get by with **GELU-only** FFNs.
- Very large LLMs tend to prefer **gated FFNs (SwiGLU/GEGLU)** because they improve **capacity per parameter** and help the network learn richer conditional behavior (different “modes” depending on the token and prompt).

---

## Layer Normalization

---

### Formula

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma}\gamma + \beta
$$

Where:

| Symbol | Meaning |
|---------|----------|
| $\mu$ | Mean of features in $x$ |
| $\sigma$ | Standard deviation of features |
| $\gamma, \beta$ | Learned scale and shift |
| $x$ | Token feature vector |

Purpose:

- Normalize each token’s feature distribution.
- Stabilizes gradients.
- Enables deeper stacks to train reliably.

You can think of LayerNorm as keeping every token’s vector in a **well-behaved range**, so that:

- Activations do not blow up or collapse as they move through dozens of blocks.
- The optimization landscape is smoother, which helps large-batch, large-model training converge.

---

### Pre-LN vs Post-LN

**Post-LN (old GPT-2)**

$$
y = \text{LayerNorm}(x + \text{Block}(x))
$$

Can destabilize when stacking many layers.

**Pre-LN (modern LLMs)**

$$
y = x + \text{Block}(\text{LayerNorm}(x))
$$

- Normalizes *before* the block.
- Improves gradient flow.
- Used in GPT-3/4, LLaMA, Mistral, etc.

---

## Residual (Skip) Connections

---

### Equation

$$
y = x + F(x)
$$

Where:

| Symbol | Meaning |
|---------|----------|
| $x$ | Input |
| $F(x)$ | Block transformation |
| $y$ | Output |

Residuals mean:  
> Don’t replace information — **refine** it.

If $F(x)$ ≈ 0, the network just passes input through → stabilizes deep training.

Gradient also has direct path:  
$$
\frac{dL}{dx} = \frac{dL}{dy}(I + F'(x))
$$

→ helps prevent vanishing gradients.

Intuitively:

- Even if the internal transformation $F$ is poorly behaved in some layers, the identity path lets gradients **flow straight through** many blocks.
- This is what allows Transformers to stack **dozens or hundreds of layers** without collapsing.

---

## Dropout & Regularization

---

### Formula

$$
h' = \frac{m \odot h}{1 - p}
$$

| Symbol | Meaning |
|---------|----------|
| $m$ | Binary mask (0 with prob $p$) |
| $h$ | Activation vector |
| $p$ | Dropout rate |
| $\odot$ | Element-wise multiply |

Keeps training robust by zeroing random activations → prevents co-adaptation.

---

### Where dropout appears

- **Attention dropout** — on attention weights before $V$  
- **FFN dropout** — after activation in FFN  
- **Residual dropout** — before residual add

Also commonly used in large LLMs:

- **Residual scaling** — use $y = x + \alpha F(x)$ with a small $\alpha$ (often decreasing with depth) so early layers cannot change the representation too aggressively.
- **Stochastic depth** — randomly skip whole blocks during training, which:
  - Makes the model behave like an **ensemble of shallower networks**.
  - Prevents over-reliance on any single block.

---

## Full Transformer Block (Decoder-only, Pre-LN)

---

$$
h_1 = x + \text{Dropout}(\text{MHA}(\text{LayerNorm}(x)))
$$
$$
h_2 = h_1 + \text{Dropout}(\text{FFN}(\text{LayerNorm}(h_1)))
$$

Where:

- **LayerNorm** → stabilizes input  
- **MHA** → inter-token communication  
- **FFN** → per-token computation  
- **Dropout** → regularization  
- **Residuals** → preserve information flow

At the top → **final LayerNorm**, then project to **vocabulary logits**.

---

## Putting It All Together — Full Transformer Flow

---

1. **Tokenize** input → token IDs  
2. Convert to **embeddings**
3. Add or apply **positional/rotary encodings**
4. Pass through **N decoder blocks:**
   - Pre-LN → MHA → residual  
   - Pre-LN → FFN → residual
5. Apply **final LayerNorm**
6. Project to **vocab logits**
7. Use **softmax** to get next-token probabilities

---

## Summary: The Modern LLM Transformer

---

| Component | Purpose |
|------------|----------|
| **Embedding + Positional Encoding** | Encode input tokens and their positions |
| **Self-Attention (MHA)** | Global token-to-token communication |
| **Feed-Forward (FFN)** | Nonlinear computation per token |
| **LayerNorm** | Stabilize and normalize activations |
| **Residuals** | Maintain signal, stabilize gradient flow |
| **Dropout** | Regularize model |
| **Stacking Blocks** | Build hierarchical understanding |

---

### Final One-Liner

> **Transformers** convert sequences into contextualized token representations through attention and residual computation — enabling scalable, parallel, and general reasoning at the heart of modern LLMs.


