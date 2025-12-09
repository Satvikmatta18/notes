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

### Step-by-step math view of MHA

1. **Start from token representations**  
   Let $x \in \mathbb{R}^{n \times d_{\text{model}}}$, where each row $x_t$ is the embedding of token $t$.

2. **Per-head linear projections**  
   For head $i$ you have:
   - $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$  
   - $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$  
   - $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$  
   Then:
   $$
   Q_i = x W_i^Q \in \mathbb{R}^{n \times d_k},\quad
   K_i = x W_i^K \in \mathbb{R}^{n \times d_k},\quad
   V_i = x W_i^V \in \mathbb{R}^{n \times d_v}
   $$

3. **Self-attention inside each head**  
   For head $i$:
   $$
   A_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}
   $$
   - Row $t$ of $A_i$ is the **attention distribution** of token $t$ over all tokens.  
   - Then the output of the head is:
   $$
   \text{head}_i = A_i V_i \in \mathbb{R}^{n \times d_v}
   $$

4. **Concatenation and mixing heads**  
   - Stack all heads:
     $$
     H = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \in \mathbb{R}^{n \times (h d_v)}
     $$
   - Mix them back into model space:
     $$
     \text{MHA}(x) = H W_O,\quad W_O \in \mathbb{R}^{(h d_v) \times d_{\text{model}}}
     $$

5. **Tiny dimensional example (no numbers, just shapes)**  
   Suppose:
   - $d_{\text{model}} = 512$, $h = 8$ heads  
   - Then $d_k = d_v = 64$ (so $8 \times 64 = 512$)  
   Each token:
   - Starts as a 512-D vector  
   - Gets mapped to 8 different 64-D subspaces (one per head)  
   - Each head does its own self-attention, then all 8 outputs are concatenated back to 512-D  
   - $W_O$ mixes these 8 “views” into the next-layer representation

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

#### How RoPE changes the dot product (relative position math)

Consider 2D for simplicity. Let $q, k \in \mathbb{R}^2$ be the **base** query/key vectors for some token content (ignoring position).  
RoPE applies rotations depending on token positions $i, j$:

$$
\tilde{q}_i = R_{\theta_i} q,\quad \tilde{k}_j = R_{\theta_j} k
$$

Now look at their dot product:

$$
\tilde{q}_i^\top \tilde{k}_j
= (R_{\theta_i} q)^\top (R_{\theta_j} k)
= q^\top R_{\theta_i}^\top R_{\theta_j} k
$$

Because rotations compose and $R_{\theta_i}^\top = R_{-\theta_i}$:

$$
R_{\theta_i}^\top R_{\theta_j} = R_{\theta_j - \theta_i}
$$

So:

$$
\tilde{q}_i^\top \tilde{k}_j = q^\top R_{\theta_j - \theta_i} k
$$

**Key point:** the attention score depends on the **angle difference** $(\theta_j - \theta_i)$, i.e. the **relative position** between tokens, not just their absolute indices. In higher dimensions, RoPE applies this pairwise to many 2D subspaces, but the same idea holds.

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

### Step-by-step FFN computation

1. **Input and dimensions**  
   - Let $x \in \mathbb{R}^{d_{\text{model}}}$ be a **single token vector**.  
   - Typical choice: hidden dimension $d_{\text{ff}} \approx 4 \cdot d_{\text{model}}$.

2. **First linear layer (expansion)**  
   - $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $b_1 \in \mathbb{R}^{d_{\text{ff}}}$  
   - Compute:
     $$
     h = W_1 x + b_1 \in \mathbb{R}^{d_{\text{ff}}}
     $$
   - This creates $d_{\text{ff}}$ different **features** as linear combinations of the original coordinates in $x$.

3. **Nonlinearity**  
   - Apply element-wise:
     $$
     u = \phi(h) \in \mathbb{R}^{d_{\text{ff}}}
     $$
   - $\phi$ could be GELU, Swish, or a gated variant (e.g. SwiGLU) which further **gates** components.

4. **Second linear layer (compression)**  
   - $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d_{\text{model}}}$  
   - Compute:
     $$
     y = W_2 u + b_2 \in \mathbb{R}^{d_{\text{model}}}
     $$
   - Now you’re back in the same dimensionality as the input token, but after a **nonlinear transformation** in a higher-dimensional space.

5. **Toy numeric sketch (very small dims)**  
   - Let $d_{\text{model}} = 2$, $d_{\text{ff}} = 4$.  
   - $x \in \mathbb{R}^2$ → multiply by $W_1$ (shape $4 \times 2$) → $h \in \mathbb{R}^4$.  
   - Apply $\phi$ → $u \in \mathbb{R}^4$.  
   - Multiply by $W_2$ (shape $2 \times 4$) → $y \in \mathbb{R}^2$.  
   Even in this tiny case, you let the network build 4 intermediate features, apply a nonlinearity, then recombine them into 2 output features.

---

## Common Activations in FFNs

---

### 1. ReLU → GELU (early Transformers)

**ReLU**

$$
\text{ReLU}(z) = \max(0, z)
$$

- **Piecewise definition (explicit):**
  $$
  \text{ReLU}(z) =
  \begin{cases}
  0, & z \le 0 \\
  z, & z > 0
  \end{cases}
  $$
- **Derivative:**
  $$
  \text{ReLU}'(z) =
  \begin{cases}
  0, & z < 0 \\
  1, & z > 0
  \end{cases}
  $$
  (undefined exactly at \(z = 0\), but set to 0 or 1 in practice).
- **Effect:** all **negative inputs are clamped to 0** and stop contributing gradients; positive inputs pass through unchanged.

Tiny numeric sketch:

- \(z = -2 \Rightarrow \text{ReLU}(z) = 0\)  
- \(z = -0.1 \Rightarrow 0\) (small negatives completely killed)  
- \(z = 0.5 \Rightarrow 0.5\), \(z = 3 \Rightarrow 3\) (acts like identity)

**GELU (Gaussian Error Linear Unit)**

Definition (conceptual):

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

where \(\Phi(z)\) is the CDF of a standard normal \(\mathcal{N}(0,1)\).

Smooth approximation used in practice:

$$
\text{GELU}(z) \approx 0.5z\!\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)
$$

How it works internally:

- Think of \(\Phi(z)\) as a **soft gate** in \([0,1]\) that increases with \(z\):
  - Very negative \(z\) → \(\Phi(z) \approx 0\) → output near 0  
  - Very positive \(z\) → \(\Phi(z) \approx 1\) → output near \(z\)
- So each component of \(z\) is scaled by **how likely a standard normal variable is to be less than \(z\)**.

Derivative (using PDF \(\phi(z)\) of \(\mathcal{N}(0,1)\)):

$$
\frac{d}{dz}\text{GELU}(z)
= \Phi(z) + z\phi(z)
$$

- \(\phi(z)\) is largest near 0 and decays for large \(|z|\), so the extra term \(z\phi(z)\) only significantly changes the gradient **around the origin**.
- This makes GELU a **smooth “soft ReLU”**: it gradually turns on units instead of snapping from 0 to 1 like ReLU.

Tiny numeric sketch (using the \(z \cdot \Phi(z)\) view, approximate):

- \(z = -2\): \(\Phi(-2) \approx 0.023\) → \(\text{GELU}(z) \approx -2 \cdot 0.023 \approx -0.046\) (small negative, not fully 0)  
- \(z = 0\): \(\Phi(0) = 0.5\) → \(\text{GELU}(0) = 0\) but derivative \(\approx 0.5\) (half-open gate)  
- \(z = 2\): \(\Phi(2) \approx 0.977\) → \(\text{GELU}(2) \approx 1.95\) (almost identity)

---

### 2. Swish / SiLU (used in LLMs)

$$
\text{Swish}(z) = z \cdot \sigma(z), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

How it works internally:

- You can see Swish as **input × sigmoid gate**:
  - \(\sigma(z)\) is in \((0,1)\) and increases with \(z\).  
  - For each component, the network learns both **the value** (\(z\)) and **how open the gate is** (\(\sigma(z)\)).

Derivative:

$$
\frac{d}{dz}\text{Swish}(z)
= \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z))
$$

- The first term \(\sigma(z)\) ensures gradients are **never identically zero** (even for negative \(z\)).  
- The second term \(z \sigma(z)(1-\sigma(z))\) peaks around where \(\sigma(z)\) is near 0.5 (around \(z \approx 0\)), making the function **slightly non-monotonic** near zero.

Shape intuition:

- For **large positive** \(z\): \(\sigma(z) \to 1\) → \(\text{Swish}(z) \approx z\), derivative \(\approx 1\).  
- For **large negative** \(z\): \(\sigma(z) \to 0\) → \(\text{Swish}(z) \approx 0\), derivative \(\approx 0\) but with a **smooth tail** (no hard corner).  
- Around \(z \approx -1 \dots 1\): the product with \(\sigma(z)\) and its derivative creates a **bump** where slightly negative values can produce slightly higher outputs than some small positive values → this is the **non-monotonic** region that gives Swish extra expressivity.

Tiny numeric sketch (approximate):

- \(z = -2\): \(\sigma(-2) \approx 0.12\) → \(\text{Swish}(-2) \approx -0.24\) (not fully zeroed)  
- \(z = 0\): \(\sigma(0) = 0.5\) → \(\text{Swish}(0) = 0\), derivative \(\approx 0.5\)  
- \(z = 2\): \(\sigma(2) \approx 0.88\) → \(\text{Swish}(2) \approx 1.76\) (close to identity)

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

### Step-by-step LayerNorm math

Let $x \in \mathbb{R}^{d}$ be the features of a single token (e.g. $d = d_{\text{model}}$).

1. **Compute mean and variance across features**
   $$
   \mu = \frac{1}{d} \sum_{k=1}^{d} x_k
   $$
   $$
   \sigma^2 = \frac{1}{d} \sum_{k=1}^{d} (x_k - \mu)^2
   $$
   In practice, a small $\varepsilon$ is added inside the square root: $\sqrt{\sigma^2 + \varepsilon}$.

2. **Normalize**
   $$
   \hat{x}_k = \frac{x_k - \mu}{\sqrt{\sigma^2 + \varepsilon}}
   $$
   Now $\hat{x}$ has mean $\approx 0$ and variance $\approx 1$ across its features.

3. **Scale and shift (learned)**
   $$
   y_k = \gamma_k \hat{x}_k + \beta_k
   $$
   - $\gamma, \beta \in \mathbb{R}^{d}$ are learned per-feature.  
   - The model can recover any needed scale/offset while still benefiting from **normalized pre-activations**.

4. **Tiny example**
   - Suppose $x = [2, 4, 6]$. Then $\mu = 4$, $\sigma^2 = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{8}{3}$.  
   - Subtract mean: $[-2, 0, 2]$; divide by $\sqrt{8/3}$ to get normalized values.  
   - If $\gamma = [1,1,1]$, $\beta = [0,0,0]$, you just have the pure normalized vector; other $\gamma,\beta$ learn useful rescalings.

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

### Step-by-step dropout mechanics

1. **Sample mask**
   - For each component $h_k$, sample $m_k \sim \text{Bernoulli}(1 - p)$:
     - $m_k = 1$ with probability $1-p$ (keep)
     - $m_k = 0$ with probability $p$ (drop)

2. **Apply mask**
   $$
   \tilde{h}_k = m_k \cdot h_k
   $$
   Some components become exactly zero.

3. **Scale to keep expectation constant**
   - During training, divide by $(1-p)$:
     $$
     h'_k = \frac{\tilde{h}_k}{1-p}
     $$
   - This ensures $\mathbb{E}[h'_k] = h_k$, so at test time you can **turn dropout off** without changing the expected magnitude of activations.

4. **Toy example**
   - Let $h = [1, 2, 3]$, $p = 0.5$.  
   - Suppose mask $m = [1, 0, 1]$ is sampled.  
   - Then $\tilde{h} = [1, 0, 3]$, and dividing by $1-p = 0.5$ gives $h' = [2, 0, 6]$.  
   - On a different training step you’d get a different mask → the network cannot rely on any single neuron always being present.

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

### Step-by-step flow through one decoder block

1. **Normalize + attend**  
   - Take input $x$.  
   - Apply LayerNorm: $\hat{x} = \text{LayerNorm}(x)$.  
   - Run MHA: $a = \text{MHA}(\hat{x})$.  
   - Apply dropout: $\tilde{a} = \text{Dropout}(a)$.  
   - Add residual: $h_1 = x + \tilde{a}$.

2. **Normalize + FFN**  
   - Apply LayerNorm again: $\hat{h}_1 = \text{LayerNorm}(h_1)$.  
   - Run FFN: $f = \text{FFN}(\hat{h}_1)$.  
   - Apply dropout: $\tilde{f} = \text{Dropout}(f)$.  
   - Add residual: $h_2 = h_1 + \tilde{f}$.

3. **Stack many such blocks**  
   - Each block refines representations **a little**, while residuals ensure information and gradients can flow through **dozens/hundreds** of layers without collapsing.

---

## Putting It All Together

---

Conceptually, a decoder-only Transformer defines a function

> “Given all previous tokens, return a probability distribution over the **next** token.”

You can write this as:

$$
p_\theta(t_{n+1} \mid t_{\le n}) = \text{softmax}(z_n), \quad
z = H W_{\text{vocab}}^\top
$$

Where:

- $t_1, \dots, t_n$ are token IDs  
- $H \in \mathbb{R}^{n \times d_{\text{model}}}$ is the final hidden states matrix (one row per token)  
- $z_n$ is the **last row** of $z$ (logits for the next token)  
- $W_{\text{vocab}}$ is the output embedding / vocab projection

Full flow end-to-end:

1. **Tokenize** input → token IDs $(t_1, \dots, t_n)$  
2. Convert to **embeddings** using a matrix $E$: $X_0 = E[t_1, \dots, t_n]$  
3. Add or apply **positional/rotary encodings** to $X_0$  
4. Pass through **$N$ decoder blocks:**
   - Pre-LN → MHA → residual  
   - Pre-LN → FFN → residual  
   to get $X_N$ (same shape as $X_0$)
5. Apply **final LayerNorm**: $H = \text{LayerNorm}(X_N)$  
6. Project to **vocab logits**: $z = H W_{\text{vocab}}^\top$  
7. Use **softmax** on the last position $z_n$ to get next-token probabilities

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


