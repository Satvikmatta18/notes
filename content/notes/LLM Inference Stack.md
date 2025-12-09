---
title: LLM Inference Stack
tags: [notes, llm, systems, inference]
---

## LLM Inference Stack

Modern LLMs are usually **trained once** but **served millions or billions of times**.  
Inference is where you spend real money and where users actually feel latency, so small
architectural and systems choices here translate directly into **cost, throughput, and UX**.

This note is a **systems‑level map** of how inference works end‑to‑end — from **raw GPU
hardware and kernels** up through **model logic, engines, serving, and profiling** — so you
can reason about bottlenecks. 

Think of the stack as layers:

1. **Hardware + Memory Substrate**  
2. **GPU Kernels & Compute Primitives**  
3. **Model Logic (Transformer forward pass)**  
4. **Graph Optimization & Compilation**  
5. **Runtime / Engine Layer**  
6. **Serving & Orchestration**  
7. **Profiling, Benchmarking, and Design Patterns**

Each higher layer is “just” a structured way of driving the layers below it.

## Layer 1 — Hardware & Memory

### GPU Memory Hierarchy

As you go **farther from the compute cores**, memory becomes **bigger but slower**. Inference performance is mostly about **how efficiently you move data through this pyramid**.

| Level | Location | Capacity (typical) | Approx Speed | Used For |
|-------|----------|--------------------|--------------|----------|
| **Registers** | Per thread | Kilobytes per SM | Fastest (~20 TB/s) | A few scalars while a kernel runs |
| **Shared Memory / L1** | On each SM | ~100 KB per SM | Very fast (~20 TB/s) | Reusing small tiles of tensors |
| **L2 Cache** | Shared across SMs | Tens of MB | High (~6 TB/s) | Passing data between blocks |
| **HBM (VRAM)** | On GPU package | 40–80 GB | 1–3 TB/s | Model weights, [KV cache](#kv-cache-size-scaling), activations |
| **CPU DRAM** | Host RAM | 100s of GB | ~100 GB/s | Models not yet loaded, staging |
| **Disk / Network** | SSD / NVMe / Ethernet | TB–PB | MB–GB/s | Loading checkpoints, remote weights |

The key optimization question: **how many times can I reuse each value after I pull it from HBM into on‑chip memory**?

### Compute Units

- GPUs have many **Streaming Multiprocessors (SMs)**. Each SM contains:
  - Scalar cores (CUDA cores) for FP32/FP16 math  
  - **Tensor Cores** for matrix‑multiply‑accumulate (MMA)  
  - Registers and shared memory  
- A kernel launch spawns **thousands of threads**, grouped into:
  - **Warps**: 32 threads executing the same instruction (SIMT)  
  - **Blocks**: many warps scheduled together on one SM  
- The SM **scheduler hides latency** by quickly switching between warps:
  - While one warp waits on memory, another warp runs compute.

Good kernels keep **SMs busy > 80%** of the time with enough warps in flight.

### Arithmetic Intensity (AI)

To understand if a kernel is **compute‑limited** or **memory‑limited**, use **arithmetic intensity**:

$$
\text{AI} = \frac{\text{FLOPs performed}}{\text{Bytes moved from HBM}}
$$

Example:

- **Matmul**: each value is reused many times → **high AI** → usually compute‑bound.  
- **KV‑cache lookup**: read lots of data, do few FLOPs → **low AI** → memory‑bound.

For an A100 (rough ballpark):

- Peak compute (FP16): **~312 TFLOPs**  
- Peak bandwidth: **~1.5 TB/s**

Theoretical “roof” AI:

$$
\frac{312 \times 10^{12}}{1.5 \times 10^{12}}
\approx 200\ \text{FLOPs/byte}
$$

- If your kernel has **AI < ~200 FLOPs/byte**, it **cannot** hit peak TFLOPs → **memory‑bound**.  
- If AI is way above this and you’re still slow, you’re limited by **compute / scheduling**.

### Roofline Model

- **X‑axis**: Arithmetic Intensity (FLOPs/byte)  
- **Y‑axis**: Achieved FLOPs/s
- Two ceilings:
  - **Bandwidth ceiling**: slanted line with slope = memory bandwidth  
  - **Compute ceiling**: horizontal line at peak FLOPs

Where a kernel lands:

- On **slanted line** → **memory‑bound** (e.g. large KV‑cache lookups during decode).  
- On **flat top** → **compute‑bound** (e.g. big prefill matmuls).  
- Far **below both** → launch/latency limited (many tiny, unfused kernels).

### Where Model State Lives in Memory

| Component | When Stored | Scales With | Notes |
|-----------|------------|------------|-------|
| **Weights** | Once per loaded model | \(\#\text{params} \times \text{bytes}\) | Fixed size; can be sharded across GPUs |
| **Activations** | Temporary, per layer | batch × seq_len | Needed for training, smaller for inference |
| **[KV cache](#kv-cache-size-scaling)** | Grows during decode | layers × heads × seq_len × head_dim × batch | Dominant memory for long contexts |
| **Optimizer states** | Training only | ~2–4× weights | Not used in pure inference |

Quick sanity check (order‑of‑magnitude):

- LLaMA‑7B, FP16 → ~14 GB weights (or ~26 GB for 13B).  
- KV cache can easily add **multiple GB** for long contexts and batches.

### Prefill vs Decode (Memory vs Compute)

- **Prefill (prompt processing)**:
  - Process the whole prompt at once.
  - Huge matmuls over \([B, T_{\text{in}}, D]\) tensors.
  - **High AI** → usually **compute‑bound**, saturates Tensor Cores.

- **Decode (autoregressive generation)**:
  - One new token per step.
  - Each step reads **all past K/V** from HBM.
  - FLOPs are small, memory reads huge → **low AI**, strongly **memory‑bound**.

Most “fancy tricks” ([KV quantization](#quantized-kv-attention), grouped/[multi‑query attention](#multi-query-attention-mqa), [continuous batching](#batching-strategies), [FlashAttention](#flashattention-example)…) are ways to **reduce memory traffic** in this regime.

### Bandwidth vs Latency

- **Bandwidth**: how much data per second you can move (GB/s).  
- **Latency**: how long each individual read/write takes (ns).  
- GPUs hide latency by:
  - **Over‑subscribing warps** (many warps ready per SM).  
  - Switching to another warp while the current one waits on memory.

Kernel design goal: **overlap memory access and compute** so Tensor Cores are almost never idle.

### Putting It Together (Layer 1)

During inference:

1. Weights and KV cache sit mostly in **HBM**.  
2. Kernels pull small **tiles** into **shared memory / registers**.  
3. Tensor Cores do MMA on those tiles, then write partial results back to HBM.  
4. Next kernels read those results, repeat.  
5. During decode, **KV cache reads dominate** memory traffic.

Anything that **reduces HBM bytes** or **increases on‑chip reuse** pushes you up toward the compute roofline.

## Layer 2 — GPU Kernels & Compute Primitives

Every neural net op (matmul, softmax, GELU, layernorm, etc.) becomes **one or more GPU kernels**.

### Kernels as Tiny Programs

- A **kernel** is a small parallel program run by thousands of GPU threads.  
- In one inference run:
  - The **model graph** (from PyTorch / TensorRT / vLLM) is decomposed into **operators**.  
  - Each operator maps to **one or more kernels**.  
  - Kernels run in sequence, reading inputs from HBM, doing math in registers, writing outputs back.

### Kernel Types

| Kernel Type | Example Ops | Core Idea | Memory Pattern |
|------------|-------------|-----------|----------------|
| **Matmul (GEMM)** | \(QK^\top\), \(A \times B\), MLP linear layers | Huge batched matrix multiplies | Reuse tiles heavily via registers + shared mem |
| **Elementwise** | add, bias, GELU, Swish, norm scale/shift | One simple op per element | HBM heavy if unfused |
| **Reduction** | softmax, layernorm, sums/max | Aggregate many values | Needs synchronization + shared memory |
| **Memory layout / copy** | transpose, pack/unpack KV cache | Change tensor shape/layout | Moves data between HBM and L2 |
| **Fused kernels** | matmul + bias + activation | Combine ops into one kernel | Less launch overhead, higher AI |

### Example: Matmul vs KV Lookup

- Matmul: `A(4096 × 4096) * B(4096 × 4096)`  
  - FLOPs ≈ `2 * 4096^3 ≈ 1.37e11`.  
  - Data (FP16, very rough): about **400 MB** moved.  
  - AI ≈ **340 FLOPs/byte** → **compute‑bound**.

- KV cache lookup:
  - Fetch a few MB of K/V, do only **kiloFLOPs**.  
  - AI ≪ 1 → extremely **memory‑bound**.

### Tensor Cores

- Specialized units for **matrix‑multiply‑accumulate** on small tiles:

  $$
  D = A \times B + C
  $$

- For FP16/BF16/FP8 they can perform many FMA operations per cycle.  
- Modern GPUs (A100, H100) have **hundreds** of Tensor Cores → **tens or hundreds of TFLOPs** peak.  
- The main job of kernel design is to **keep Tensor Cores fed**:
  - Tile inputs.
  - Use shared memory.  
  - Reuse tiles before evicting them back to HBM.

### Kernel Fusion

Every kernel launch has overhead (~10–50 µs). A single Transformer layer might naively require **100+ tiny kernels** for activations, bias adds, norms, and residuals.

Fusion idea: **combine consecutive ops that touch the same data**.

Examples:

- `y = GELU(Wx + b)`  
  - Naive: `matmul` → `bias add` → `GELU` (3 kernels).  
  - Fused: **1 kernel** that does all three while data stays in registers.

- Attention:
  - Naive: matmul \(QK^\top\) → scale → mask → softmax → matmul with \(V\).  
  - Fused (e.g. FlashAttention): a **single tiled kernel** that never materializes the full \(T \times T\) score matrix in HBM.

Benefits:

- Fewer kernel launches → less CPU overhead.  
- Fewer HBM trips → higher AI and throughput.  

### Precision and Data Movement

| Format | Bytes / value | Typical Use | Effect |
|--------|----------------|------------|--------|
| **FP32** | 4 | Training / reference | High accuracy, slowest |
| **FP16 / BF16** | 2 | Standard inference | Good balance of speed & accuracy |
| **FP8** | 1 | TensorRT‑LLM, Hopper | 1.8–3× speedup, smaller memory |
| **INT8 / INT4** | 1 / 0.5 | Quantized engines | Big HBM savings, some accuracy cost |

Moving data is often the real cost. **Halving precision halves HBM traffic**, but kernels must handle:

- **Scaling and dequantization** internally.  
- Often fused into the matmul/attention kernels so overhead is tiny.

### CUDA Graphs

- Instead of launching hundreds of kernels individually:
  - Record the entire forward pass once as a **CUDA Graph**.  
  - Replay it as a single “graph launch” each inference.
- This removes per‑kernel CPU launch overhead and stabilizes latency.  
- Used heavily in TensorRT‑LLM and similar engines for **prefill**.

### Memory Optimizations Inside Kernels

Common techniques to increase AI and avoid HBM bottlenecks:

- **Tiling**: load submatrices into shared memory, reuse many times.  
- **Double buffering**: overlap compute on tile \(n\) with loading tile \(n+1\).  
- **Asynchronous copy** (`cp.async` on Hopper): move data without stalling warps.  
- **Persistent kernels**: keep blocks alive across many tokens instead of relaunching every step.

#### FlashAttention Example

Naive attention:

- Compute scores = \(QK^\top\)  
- Apply softmax  
- Multiply by \(V\)  
- Needs \(O(T^2)\) memory and re‑reads \(K, V\) many times.

FlashAttention:

- Tiles \(Q, K, V\) into shared memory blocks.  
- Computes softmax **incrementally per tile**, tracking running max and sum.  
- Writes outputs once.  
- Same \(O(T^2)\) math, but **much less HBM traffic** → higher AI, 2–4× faster on long sequences.

## Layer 3: Model Logic (Transformer Forward Pass)

This is the **math blueprint** of the model: what operations to perform and in what order for every token. Layers 1–2 just execute this blueprint efficiently.

### Decoder‑Only Transformer Block

At a high level (per layer):

1. **LayerNorm / RMSNorm**  
2. **Self‑Attention** (with KV cache) + residual  
3. **LayerNorm / RMSNorm**  
4. **Feed‑Forward Network (MLP)** + residual

Repeat this block **30–70×** depending on model size.

### Tokens → Vectors → Matrices

- Text prompt → token IDs via tokenizer.  
- Each token ID → embedding vector \(e \in \mathbb{R}^{d_{\text{model}}}\).  
- A batch of tokens forms:

$$
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

During inference this \(X\) passes through every Transformer layer.

### Self‑Attention (Recap at Model Logic Level)

Given current hidden states \(X\):

1. **Linear projections**:
   $$
   Q = X W_Q,\quad K = X W_K,\quad V = X W_V
   $$
   Each has shape \([B, T, H, d_{\text{head}}]\) after reshaping into heads.

2. **Scaled dot‑product attention**:
   $$
   \text{Scores} = \frac{QK^\top}{\sqrt{d_{\text{head}}}}
   $$

3. **Mask + softmax**:
   - Apply **causal mask** so tokens can’t see the future.  
   - Then:
     $$
     \text{Attn} = \text{softmax}(\text{Scores})
     $$

4. **Weighted sum of values**:
   $$
   \text{Context} = \text{Attn} \cdot V
   $$

5. **Concat + output projection**:
   $$
   \text{Out} = \text{Concat}(\text{Context heads}) W_O
   $$

6. **Residual add**:
   $$
   X \leftarrow X + \text{Out}
   $$

### Feed‑Forward Network (MLP)

Per token:

1. First linear:
   $$
   z_1 = W_1 x + b_1
   $$
2. Activation (e.g. GELU, SwiGLU):
   $$
   h = \phi(z_1)
   $$
3. Second linear:
   $$
   z_2 = W_2 h + b_2
   $$
4. Residual:
   $$
   x \leftarrow x + z_2
   $$

SwiGLU variant:

$$
\text{SwiGLU}(z_1, z_2) = \text{Swish}(z_1) \odot z_2
$$

This often improves training efficiency and works well with FP8 quantization.

### Normalization & Residuals

- **LayerNorm / RMSNorm** keep activations numerically stable.  
- **Residual connections** preserve signal and gradient flow in deep stacks.  
- At inference these are “just” elementwise ops, but they generate many small kernels → good targets for **fusion**.

### Prefill vs Decode at Model Level

- **Prefill phase**:
  - Process entire input sequence at once.  
  - Build K/V for every position and layer.  
  - Complexity \(\mathcal{O}(T^2)\) for attention matmuls.  
  - High AI → compute‑bound.

- **Decode phase**:
  - Only the **new token** passes through the layer each step.  
  - Reuse stored K and V from previous steps (KV cache).  
  - Complexity **O(T)** per token but dominated by memory reads.

### Attention Variants (MHA, MQA, GQA, Flash, Sparse)

#### Multi‑Head Attention (MHA)

- Each head has its **own** $W_Q, W_K, W_V$.  
- Very flexible, high quality, but:
  - KV cache memory is roughly `2 * N_layers * N_heads * T * d_head * bytes`.  
  - Many HBM reads during decode.

#### Multi‑Query Attention (MQA)

- Many **query heads**, but **single shared** \(K, V\) across heads.  
- KV cache:
  - Instead of \([N_{\text{heads}} \times T \times d_{\text{head}}]\), store \([1 \times T \times d_{\text{head}}]\).  
  - KV memory drops by \(\approx 1 / N_{\text{heads}}\).
- Slight quality drop; used where memory is the main bottleneck.

#### Grouped‑Query Attention (GQA)

- Middle ground: group multiple query heads per shared K/V group.  
- Example: 8 query heads per K/V group → **8× less memory** than full MHA but more variety than MQA.  
- Exploits redundancy: many heads tend to look at similar positions.

#### FlashAttention

- Same math as standard attention.  
- Tiled implementation that:
  - Never materializes full \(T \times T\) scores in HBM.  
  - Keeps work in shared memory.  
  - Increases AI and reduces memory by a large factor (2–4× speedup on long prompts).

#### Sliding‑Window / Sparse Attention

- Each token attends only to recent **window** `[t-w, ..., t]`.  
- Complexity **O(T · w)** vs **O(T²)**.  
- Great for very long contexts with local structure; small quality tradeoff for tasks needing long‑range dependencies.

#### Quantized KV Attention

- Store K/V in reduced precision (FP8/INT8):  
  $$
  K_q = s_K \cdot \text{dequant}(K_{\text{int8}}), \quad
  V_q = s_V \cdot \text{dequant}(V_{\text{int8}})
  $$
- Run normal attention on dequantized values.  
- KV cache halves in size → faster memory‑bound decode with minimal quality loss.

### KV Cache Size Scaling

During **decode**, every attention layer stores the **keys and values** for all tokens
seen so far. This is the **KV cache**:

- For each new token, each head in each layer appends one key vector and one value vector.  
- Those K/V rows live in **HBM** and are reused on every subsequent step so we don’t
  recompute attention over the whole prefix.  
- The term “size scaling” here is about **how that cache grows** as you increase the
  number of layers, heads, sequence length, batch size, or change precision.

Let:

- `N_layers`: number of layers  
- `N_heads`: attention heads per layer  
- `d_head`: head dimension  
- `T`: sequence length  
- `B`: batch size  
- `bytes`: bytes per value (2 for FP16)

Then:

- `KV_size = 2 * N_layers * N_heads * d_head * T * bytes * B`

Example (rough):  
`N_layers=32, N_heads=32, d_head=128, T=2048, B=8, bytes=2` → multiple GB of KV.

For very large models, **MQA / GQA + KV quantization** are mandatory to fit in 40–80 GB GPUs.

### Model Ops → Kernel Types → Optimizations

It’s useful to map each Transformer sub‑op to **kernel type** and **typical optimization**:

| Model Operation | Kernel Type | Typical Optimization |
|-----------------|------------|----------------------|
| **Q, K, V projections** | GEMM (matmul) | Tensor Cores, FP16/FP8, fused bias |
| **Attention scores (QKᵀ)** | Matmul | FlashAttention, tiling, shared mem |
| **Softmax + masking** | Reduction + elementwise | Shared‑mem reductions, fusion with matmuls |
| **(Attn × V)** | Matmul | Fused tile kernels (FlashAttention) |
| **MLP (FFN)** | GEMM + elementwise | Fused MLP (matmul + bias + activation) |
| **LayerNorm / RMSNorm** | Reduction + elementwise | Kernel fusion, vectorized loads/stores |

These choices determine whether a layer tends to be **compute‑bound** (big GEMMs) or **memory‑bound** (KV lookups, norms, small elementwise ops).

## Layer 4: Graph Optimization & Compilation

The **computation graph** is a static map of all tensor ops: matmuls, adds, norms, activations, etc. One decoder block might look like:

> input → norm → (Q, K, V linears) → attention → residual → norm → MLP → residual → output

At runtime, this graph becomes a **schedule of GPU kernels**. Compilers rewrite this graph to minimize launches and HBM trips.

### Graph Tracing

Tools like TorchDynamo, ONNX export, TensorRT builders:

- Run your PyTorch model once while **recording every tensor op**.  
- Output a static **IR (intermediate representation)** graph:
  - Fixed shapes and dtypes (or shape ranges).  
  - All data dependencies explicit.  
  - Python control flow is gone.

This static IR is what gets optimized.

### Simplification Passes

Before heavy optimization:

- **Constant folding**: precompute anything fixed (e.g. RoPE sin/cos tables).  
- **Dead‑code elimination**: drop training‑only paths, unused outputs.  
- **Shape simplification**: collapse redundant reshapes/transposes (e.g. \(T(T(x)) \rightarrow x\)).  
- **Common subexpression elimination**: reuse identical subgraphs.

These passes alone can shrink the graph by **10–20%**.

### Operation Fusion (Graph Level)

Goal: avoid repeatedly writing to and reading from HBM.

Patterns:

- **Fused MLP**: matmul + bias + GELU → 1 kernel.  
- **Fused attention**: matmul \(QK^\top\) + scaling + masking + softmax + matmul with \(V\) → FlashAttention‑style kernels.  
- **Add + norm**: fuse into a single kernel.

Instead of ~500 kernels per layer, fusion can cut this to a few dozen → **major latency reduction**.

### Precision Lowering

Compilers decide which ops can safely run at lower precision:

- **Norms / logits** → often kept at FP16/BF16.  
- **Matmuls / MLPs / attention** → may run at FP8 or INT8.  
- Quantization inserts scale factors:
  - These casts and scales are **fused** into surrounding kernels so they disappear at runtime.

### Memory Layout Rewrites

Different kernels prefer different layouts (e.g. **NHWC vs NCHW**, head‑major vs sequence‑major).

The compiler:

- Reorders tensors offline (in weights) or once at load time.  
- Avoids expensive runtime transposes.  
- Ensures memory is **coalesced** (threads 0–31 read adjacent addresses).

### Static vs Dynamic Graphs

- **Static graphs**:
  - Shapes fixed at compile time.  
  - Maximal fusion and CUDA Graph replay.  
  - Great for large, steady workloads (TensorRT‑LLM prefill).

- **Dynamic graphs**:
  - Shapes vary at runtime.  
  - More flexible for mixed prompts and interactive traffic.  
  - Typically less fusion; used for decode in systems like vLLM.

Some engines combine both: **static prefill graphs + dynamic decode kernels**.

### Memory Planning & Buffer Reuse

The compiler tracks **tensor lifetimes** and reuses buffers:

- Once a tensor is no longer needed, its memory can hold later activations.  
- Temporary scratch buffers for fused kernels (FlashAttention, MLP) are pooled, not repeatedly `malloc`/`free`d.  
- KV pages are allocated once and reused as sequences finish.

Result: **smaller VRAM footprint**, allowing higher batch sizes.

### Emitting the Engine Plan

Compilers output a **hardware‑ready engine**:

| Engine | Output | Contains |
|--------|--------|----------|
| **TensorRT‑LLM** | `.plan` file | Fused kernels, launch order, precision config |
| **TorchInductor** | Triton / CUDA bundle | Fused kernels + small runtime driver |
| **vLLM** | Runtime IR | Paged KV ops + scheduler‑friendly layout |

This plan is what the **runtime / engine layer** actually executes.

## Layer 5: Runtime / Engine Layer

Typical stack:

> Frontend API → Scheduler / Engine Runtime → Compiled model graph → GPU kernels → Hardware

The inference engine sits **between** the API and the compiled model. It decides **when, how, and where** each inference runs given memory and latency constraints.

### Lifecycle

1. **Request arrives** at API.  
2. Engine **queues** it and possibly waits a short window to batch.  
3. **Prefill phase**:
   - Process full prompt.  
   - Compute initial KV cache.  
   - Compute first output tokens.
4. **Decode loop**:
   - Generate tokens one by one using cached KV.  
   - Stream tokens back to client.
5. Request finishes → KV cache released or reused.

### Engine Responsibilities

- **Request handling**: accept new requests, manage their states.  
- **Batching**: combine compatible requests into a single GPU call.  
- **KV cache management**: allocate, page, and free KV memory.  
- **Memory / precision management**: choose FP16/FP8/INT8 modes within VRAM limits.  
- **Decoding**: sampling, beam search, speculative decoding, grammar constraints.  
- **Scheduling**: decide which requests run each decode tick.

### Core Data Structures

**Request table** (per sequence):

| Field | Example |
|-------|---------|
| Request ID | `12345` |
| Current length | `56` tokens |
| Max length | `512` |
| KV cache pointers | `[layer][page_ptrs]` |
| Sampling config | `temperature=0.8`, top‑k, top‑p |
| Status | waiting / prefill / decoding / finished |

**Batch descriptor** (per decode tick):

```text
Batch {
  batch_size: 8
  seq_ids: [1, 5, 9, 14, 18, 22, 27, 33]
  input_tokens: [last token of each sequence]
  kv_ptrs: [pointers to each seq’s KV pages]
}
```

**KV cache map**:

```text
KV Cache:
  Layer 0: [K0, V0]
  Layer 1: [K1, V1]
  ...
  Layer N: [KN, VN]
```

During **prefill**, these are built from scratch. During **decode**, we **append one row** (new token) per step.

### Batching Strategies

- **Static batching**:
  - Collect \(N\) requests, run them as one batch until all finish.  
  - Simple, efficient if all sequences have similar lengths.  
  - Wastes compute when some finish earlier.

- **Dynamic batching**:
  - Accumulate requests over a short time window (e.g. 10–50 ms).  
  - Form a batch from whatever is available.  
  - Batch composition is fixed during a run.

- **Continuous batching** (vLLM, SGLang):
  - Maintain a **global pool** of active sequences.  
  - On each decode tick:
    - Collect all unfinished sequences.  
    - Build a batch descriptor (last tokens + KV pointers).  
    - Run one forward step, append KV, update pool.  
  - Sequences can **enter and leave** the batch at any time.  
  - Keeps GPU utilization high under irregular real‑world traffic.

### Prefill vs Decode at Runtime

| Phase | Shape | Dominant Cost | Engine Behavior |
|-------|-------|---------------|-----------------|
| **Prefill** | `[B, T_in, D]` | Compute (FLOPs) | Static batches, compiled graphs ([TensorRT](#engine-case-studies-narrative-view), [CUDA Graphs](#cuda-graphs)) |
| **Decode** | `[B, 1, D]` | Memory (HBM) | Dynamic/[continuous batching](#batching-strategies), [paged KV](#kv-cache-layouts--management), quantized KV |

### KV Cache Layouts & Management

**Naive KV cache**:

- Each sequence has its own contiguous `[T, ...]` KV tensor per layer.  
- Pros: simple indexing.  
- Cons: fragmentation, over‑allocation (must reserve max length).

**Paged KV cache** (vLLM style):

- Treat GPU memory as fixed‑size **pages** (e.g. 128 tokens).  
- Each sequence owns a **list of page pointers**:

```text
seq_A → [page_1, page_7, page_9]
seq_B → [page_3, page_8]
```

- When a token is added, allocate a new page only when needed.  
- Attention kernels (PagedAttention) iterate over page lists instead of one contiguous tensor.

Benefits:

- Less waste, easier eviction and reuse.  
- Enables large batches under strict VRAM.

**KV quantization** at runtime:

| Precision | Bytes/value | Memory Saved | Notes |
|-----------|-------------|--------------|-------|
| FP16 | 2 | – | Default |
| FP8 | 1 | 2× | TensorRT‑LLM support |
| INT8 | 1 | 2× | Needs per‑channel scales |

Quantizing KV **halves memory traffic** in decode and speeds up attention.

### Goals: TTFT and Throughput

- **TTFT (Time To First Token)**:
  - From request arrival to first token.  
  - Dominated by prefill + CPU / graph overhead.

- **TPOT (Time Per Output Token)**:
  - Roughly:
    $$
    \text{TPOT} =
    \frac{t_{\text{last}} - t_{\text{first}}}{\max(1, N_{\text{out}} - 1)}
    $$
  - Measures steady‑state decode speed.

- **Throughput**:
  - Total output tokens / wall‑clock time per GPU.  
  - Depends on **decode efficiency** and batching.

Runtime design is about trading off **TTFT vs TPOT vs utilization**.

### Advanced Runtime Tricks

Many “systems” features live here; they mostly try to keep GPUs busy during **decode**, where work is small and memory‑bound.

#### Prefix reuse & prefix trees

- If many requests share the same **prompt prefix**, you can compute that prefix **once**, cache its KV pages, and then:
  - Let new sequences **reuse** those pages until their text diverges.  
  - Only allocate new KV pages for tokens after the fork point.
- Systems like **SGLang** generalize this into a **prefix tree**:
  - Each node stores pointers to KV pages for that prefix.  
  - New sequences walk the tree, share existing pages, and allocate new pages only when they branch.

This reduces both **KV recompute** and **KV memory**, which helps decode throughput on prefix‑heavy workloads (chat, RAG with templates).

#### Prefill chaining

- Instead of launching many small prefill graphs for short prompts, engines can **chain multiple prompts** into one larger CUDA Graph capture.  
- This amortizes launch overhead and improves SM utilization for “many tiny prompts” traffic.

#### Multi‑stream execution

- Use multiple CUDA streams to overlap:
  - Stream 0 → prefill  
  - Stream 1 → decode  
  - Stream 2 → speculative verification or auxiliary work
- With careful priorities and dependencies, you can overlap **H2D copies, compute, and sampling**.

#### Speculative decoding

- Run a small **draft model** to predict \(k\) future tokens.  
- Run the large **target model** on the current prefix + those \(k\) candidates.  
- If the target’s predictions match the draft up to position \(m\): accept those \(m\) tokens, skipping \(m\) target‑model steps.  
- Otherwise, accept tokens up to the first mismatch and continue normally.
- Requires **two KV caches** (draft + target) and logic to **merge KV state** when tokens are accepted.

Throughput gains are roughly `(1 - error_rate) * k` fewer big‑model steps, at the cost of running the cheap draft model.

#### Structured / grammar‑constrained decoding

- For JSON / XML / regex grammars, the runtime applies a **logit mask** each step:
  - Disallow tokens that would violate the grammar at the current position.  
  - Renormalize probabilities and sample.
- Often fused into the sampling kernel so constraint logic doesn’t add big overhead.

### Tokenization & Decoding Algorithms

Even though tokenization and decoding sit “above” the core engine, they show up directly in
**TTFT, TPOT, and throughput**.

#### Tokenization

- Most LLMs use **BPE / SentencePiece‑style subword vocabularies**:
  - Map raw text → token IDs using a learned merge table.  
  - Trade‑off: **larger vocab** = shorter sequences but bigger logits and embedding tables;
    **smaller vocab** = longer sequences but smaller head.
- Inference implications:
  - Tokenization cost on CPU can be **non‑trivial** at very high QPS.  
  - Pre‑tokenizing common prompts or moving tokenization into a fast Rust/C++ service can
    shave a few ms off **TTFT**.  
  - Vocab size affects **logit matmul cost** (final projection) and **sampling** time.

#### Decoding algorithms

- **Greedy decoding**:
  - Always pick argmax(logits).  
  - Fastest and easiest to batch; used for deterministic APIs or internal tools.

- **Sampling (top‑k, top‑p, temperature)**:
  - Top‑k: keep largest k logits, renormalize.  
  - Top‑p: keep smallest set of tokens whose cumulative prob ≥ p.  
  - Temperature: scale logits by `1/T` before softmax.  
  - Adds a small per‑step cost but greatly improves **diversity and quality**.

- **Beam search**:
  - Maintain multiple candidate sequences (beams), expand all, keep best k by score.  
  - Roughly multiplies decode cost by beam width; often used for translation / structured
    tasks, less common for chat‑style LLMs.

Runtime placement:

- Tokenization usually lives in the **frontend / preprocessing** layer.  
- Sampling and beam search live in the **engine runtime**, often **fused** with:
  - logit computation  
  - grammar masks  
  - temperature/top‑k/top‑p truncation

## Layer 6: Serving & Orchestration

When you move to production, you stop thinking about one GPU and start thinking about **fleets of GPUs, many models, and SLAs**.

### API Frontends

- **HTTP (REST)**:
  - Easy JSON APIs, good for external clients.  
  - Higher overhead (2–5 ms per request).
- **gRPC**:
  - Binary protocol over HTTP/2.  
  - Low overhead (< 1 ms) and good for high‑QPS internal calls.

### Routing & Scheduling

- **Ingress / router**: accepts requests, handles auth.  
- **Scheduler**: decides which GPU pool / model instance handles each request.  
- **Load balancer**: sends traffic to the least‑loaded replica.

### Multi‑GPU and Multi‑Node

Modes:

- **Tensor parallelism (TP)**: split weight matrices across GPUs.  
- **Pipeline parallelism (PP)**: split layers across GPUs.  
- **Expert / MoE parallelism**: different experts on different GPUs.  
- **Sequence parallelism**: split tokens across GPUs for long‑context serving.

Serving systems combine these to:

- Fit very large models.  
- Achieve high throughput under high concurrency.

### Fine‑Tuning vs Inference

- Full SFT/DPO/PPO changes **weights** → runs on training clusters.  
- **LoRA / QLoRA** add low‑rank adapters:
  - At inference, adapters can be **loaded dynamically** per tenant/domain.  
  - Many adapters share a common base model.
- Inference systems must:
  - Manage adapter loading/unloading.  
  - Cache adapters in pinned CPU memory or GPU memory.  
  - Keep latency acceptable despite extra indirection.

Serving layers also decide **how to share KV and weights across replicas** (e.g. prefix‑KV sharing across sessions, or routing similar tenants to the same adapter‑cache‑hot GPUs).

## Layer 7: Profiling, Benchmarking & Design Patterns

Goal: explore the tradeoff space:

- **Latency** (per request, per token)  
- **Throughput** (tokens/sec/GPU)  
- **Cost** (GPU‑hours per token)  
- **Quality** (precision / quantization effects)

### Key Metrics

| Symbol | Meaning | Why It Matters |
|--------|---------|----------------|
| **TTFT** | Time to first token | Captures prefill + CPU overhead |
| **TPOT** | Time per output token | Decode speed bottleneck |
| **Throughput** | tokens/sec/GPU | Aggregate efficiency |
| **SM util %** | SM occupancy | Compute utilization |
| **HBM BW %** | % of peak bandwidth | If >80% → memory‑bound |
| **AI** | FLOPs / Bytes | Compute vs memory regime |
| **Occupancy** | Active warps / max warps | Kernel efficiency |
| **Cache hit rate** | L2 / shared reuse | Memory locality |

### Sweeps

| Sweep Type | What Changes | What You Learn |
|-----------|--------------|----------------|
| **ISL sweep** | Input sequence length | Prefill scaling, compute vs graph limits |
| **OSL sweep** | Output length | Decode scaling, KV cache bandwidth cost |
| **Concurrency sweep** | # parallel requests | Batching efficiency, scheduler behavior |
| **Engine/precision sweep** | Backend + FP16/FP8/INT8 | Accuracy vs throughput |
| **KV cache sweep** | FP16 → FP8 → INT4 | Memory vs latency balance |

### Profiling Workflow

1. **Run baseline**:
   - Fixed model / engine / ISL / OSL / batch.  
   - Collect TTFT, TPOT, SM%, HBM%, tokens/sec.

2. **Nsight Systems**:
   - View kernel timelines, CPU launch gaps, GPU stream overlap.

3. **Nsight Compute** on hot kernels (e.g. PagedAttention):
   - Measure FLOPs/s and memory throughput.  
   - Compute AI and position on roofline.

4. **Classify bottleneck**:
   - Memory‑bound: AI low, HBM% high → use FP8, KV quantization, paged attention.  
   - Compute‑bound: AI high, SM% low → improve fusion, occupancy, batch size.  
   - CPU‑bound: large launch gaps → CUDA Graphs, C++ runtimes (TensorRT, Triton).

### SLOs, Capacity Planning & Backpressure

- **SLOs** are usually expressed as **p95/p99 TTFT and TPOT** at a given concurrency and
  ISL/OSL distribution.  
- To hit them you need:
  - Enough GPU capacity (tokens/sec/GPU × number of GPUs ≥ expected load).  
  - Batching policies that don’t explode tail latency at high QPS.  
  - Admission control so the system can shed load instead of timing out everyone.

Rules of thumb:

- For a fixed model/engine, you can estimate capacity from a **decode throughput
  benchmark**:  
  `capacity_tokens_per_sec ≈ GPUs * tokens_per_sec_per_GPU`.  
- Concurrency limit for a target TPOT is roughly:  
  `max_concurrency ≈ capacity_tokens_per_sec / target_TPOT_tokens_per_sec_per_request`.  
- When queues grow too long, apply **backpressure**:
  - Reject or defer new requests.  
  - Down‑weight very long prompts or outputs.  
  - Preferentially batch short jobs together.

### Reliability & Observability

Serving infra is useless if you can’t see when it’s on fire.

- **Observability**:
  - Per‑request traces with TTFT/TPOT, ISL/OSL, engine, precision, and batch id.  
  - GPU metrics: SM%, HBM BW%, memory usage, OOMs, kernel errors.  
  - Engine‑level stats: batch sizes, decode ticks/sec, KV page usage.
- **Reliability patterns**:
  - Timeouts + retries when a node dies mid‑decode (with sensible limits).  
  - Health checks and automatic draining of sick replicas.  
  - Graceful degradation: fall back to smaller models or lower precision when under pressure.

### Searching the Inference Config Space

You can treat **engine + precision + batching + KV strategy** as a **latent space of
configs** and explore it systematically.

- Axes to sweep:
  - ISL / OSL distributions (short vs long prompts and outputs).  
  - Concurrency patterns (steady vs bursty).  
  - Engine/backend (vLLM, TensorRT‑LLM, SGLang, DeepSpeed).  
  - Precision & KV format (FP16, FP8, INT8, INT4 KV).  
  - Parallelism/sharding strategies (TP/PP, MoE on/off).
- For each point, record:
  - TTFT/TPOT p50/p95/p99, throughput, SM%/HBM%, and cost/GPU‑hour per million tokens.  
  - Operational complexity (cold‑start times, build times, stability).
- The goal is to find **Pareto‑optimal configs**: you can’t improve cost, latency, or
  throughput without hurting at least one of the others.

### Stage‑Level View: Prefill vs Decode vs Engine Runtime

You can also think of the system in **three stages**, each with its own bottlenecks and tricks:

| Stage | Workload Type | Dominant Cost | Typical Optimizations |
|-------|---------------|---------------|------------------------|
| **Prefill** | Dense, parallel over `[B, T_in, D]` | Compute (FLOPs) | FP8/INT8 matmuls, CUDA Graphs, TensorRT static plans, large static batches |
| **Decode** | Sparse, sequential over `[B, 1, D]` | Memory (HBM) | KV quantization, PagedAttention, continuous batching, prefix reuse |
| **Engine runtime** | Dynamic scheduling & bookkeeping | CPU / coordination | Dynamic/continuous batching, async streams, minimizing Python, C++ runtimes |

### Engine Comparisons (Very High Level)

| Concept / Feature | vLLM | TensorRT‑LLM | SGLang | DeepSpeed‑Inf |
|-------------------|------|--------------|--------|---------------|
| **Paged KV cache** | Core | Runtime / alloc layer | Yes + prefix tree | Yes, more sharding‑focused |
| **Continuous batching** | Yes (decode core) | In‑flight batching | Yes | Limited |
| **FP8 / INT8 core support** | External / emerging | Strong | Mixed | Mixed |
| **CUDA Graph prefill** | Not main focus | Big focus | Limited | Possible |
| **Speculative decoding** | No | Not primary | Yes | No |
| **Structured decoding** | Basic / external | Not core | Strong (JSON/grammar) | Not focus |
| **Multi‑GPU sharding** | Growing | Strong (TP/PP) | Backend‑dependent | Strong |

Broadly:

- **vLLM**: continuous batching + paged KV + custom attention kernels → strong decode throughput.  
- **TensorRT‑LLM**: heavy graph‑level and kernel‑level optimization + FP8 → near‑peak FLOPs on prefill and strong decode.  
- **SGLang**: vLLM‑style runtime plus **speculative decoding** and **prefix sharing**.  
- **DeepSpeed‑Inf**: very strong on multi‑GPU sharding and parallelism.

### Engine Case Studies (Narrative View)

**vLLM**

- Core: **continuous batching** for decode + **paged KV cache** + custom PagedAttention kernels.  
- Data structures: request table, **KV page pool**, and per‑step **decode batch descriptors** (sequence IDs, last tokens, KV page lists, position IDs, masks).  
- Flow: prefill builds KV pages and inserts sequence into an active pool → decode repeatedly forms batches from that pool and steps all sequences together → finished sequences free their pages back to the pool.

**TensorRT‑LLM**

- Build phase: imports a checkpoint, traces a mostly static graph, fuses ops into large attention/MLP blocks, chooses precision (FP16/FP8/INT8), and emits a **plan file** (engine) with kernels + launch schedule.  
- Prefill: captures the entire prefill DAG as a **CUDA Graph**, then replays it for each batch → almost no CPU dispatch overhead, near‑peak Tensor Core utilization.  
- Decode: uses paged KV buffers per sequence and in‑flight batching; fused attention kernels handle quantization and page layout.

**SGLang**

- Builds on vLLM‑style **paged KV + continuous batching**, then adds:  
  - **Speculative decoding** (draft + target model, KV merge on accept).  
  - **Prefix tree KV sharing** for common prompts.  
  - **Grammar/JSON‑constrained decoding** fused into the sampling path.

**DeepSpeed‑Inference**

- Focuses on **multi‑GPU parallelism**: tensor parallelism, pipeline parallelism, expert/MoE parallelism.  
- Integrates with custom kernels but is especially strong at **sharding large models** across many GPUs and nodes while keeping throughput high.

## Why LoRA / QLoRA Show Up in Inference Stacks

Even though **LoRA / QLoRA** are training‑time techniques, inference systems have to **support them directly** because they change *what weights you load and how you route requests*.

### LoRA — Low‑Rank Adapters

Core idea:

- Instead of fully fine‑tuning a huge matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, learn low‑rank updates:
  $$
  W' = W + \Delta W,\quad \Delta W = A B^\top
  $$
  where $A \in \mathbb{R}^{d_{\text{out}} \times r}$, $B \in \mathbb{R}^{d_{\text{in}} \times r}$, and $r \ll d_{\text{in}}, d_{\text{out}}$.
- During inference, you compute:
  $$
  y = W x + A(B^\top x)
  $$
  So one base matmul plus a small low‑rank correction.

Why inference engines care:

- **Per‑tenant / per‑task adapters**: many LoRA adapters share a single base model.  
- Inference software needs to:
  - Load/unload adapter weights on demand.  
  - Apply the extra \(A(B^\top x)\) term efficiently (often via fused kernels).  
  - Route each request to the right base model + adapter combo.

Impact on the stack:

- **Layer 3–4 (Model + Graph)**: graph includes extra low‑rank matmuls.  
- **Layer 5 (Runtime)**: must manage many small adapter weight sets and sometimes hot‑swap them between requests.  
- **Layer 6 (Serving)**: routing decisions may depend on which LoRA adapter a customer uses.

### QLoRA — Quantization‑Aware LoRA

QLoRA combines **4‑bit quantization of the base model** with **LoRA adapters trained in higher precision**.

- Base weights \(W\) are stored in very low precision (e.g. NF4/int4).  
- Adapters \(A, B\) are kept in FP16/BF16.  
- Effective weight during inference:
  $$
  W' = \text{dequant}(W_{\text{4bit}}) + A B^\top
  $$

Why inference engines support QLoRA:

- You can host **many QLoRA fine‑tunes** of a big model in the VRAM that would normally hold only **one** full‑precision copy.  
- Providers expose “bring‑your‑own‑LoRA” or “run fine‑tuned model X” APIs without loading a whole separate checkpoint.

Stack effects:

- **Layer 1–2**: more quantized matmuls (4‑bit) and small FP16 adapter matmuls.  
- **Layer 4**: compilers must fuse **dequant + base matmul + LoRA matmul** where possible.  
- **Layer 5–6**: runtime has to manage which adapter is attached to which request, and cache popular adapters.

### Why Inference Providers Expose LoRA / QLoRA

From the provider’s perspective:

- **Economics**: one base model + many tiny adapters → much higher GPU utilization and tenant density.  
- **Customization**: customers can fine‑tune behavior (domain, tone, safety rules) without shipping full checkpoints.  
- **Operational simplicity**: adapter weights are small enough to move around quickly between nodes, so you can spin up or tear down fine‑tuned variants on demand.

So even though LoRA/QLoRA are “training tricks”, **supporting adapters is now part of the core inference stack**, right next to KV cache management and quantization.

## Disaggregated Inference — From DistServe to NVIDIA Dynamo‑style Systems

Classic serving stacks assume **one engine instance = one full model** sitting on a fixed
set of GPUs. Disaggregated designs instead **split responsibilities across pools of
machines**, usually along the **prefill vs decode** boundary.

### Why disaggregate?

- Prefill is **short, bursty, compute‑bound** and likes large, fast GPUs and big batches.  
- Decode is **long‑lived, memory‑bound** and likes steady, high‑occupancy scheduling plus
  lots of KV storage.  
- If you bind them to the same GPU, you either:
  - over‑provision for rare prefill bursts, or  
  - hurt decode throughput when prompts spike.

Disaggregated systems try to fix this by **separating prefill and decode hardware** and
letting a central scheduler route work between them.

### DistServe (Hao Zhang et al.)

High‑level ideas (paper: *DistServe: Disaggregating Prefill and Decoding for LLM Serving*):

- **Two GPU pools**:
  - **Prefill workers**: optimized for big matmuls and static CUDA Graphs.  
  - **Decode workers**: optimized for continuous batching and KV‑heavy attention.
- **KV handoff**:
  - After prefill, the KV cache for each sequence is **transferred** (or re‑materialized)
    to a decode worker.  
  - Decode workers then keep the sequence for the rest of its life.
- **Scheduling**:
  - Router assigns incoming prompts to available prefill workers.  
  - When prefill finishes, it enqueues the sequence into the decode cluster, which runs a
    vLLM‑style continuous batching loop.

Benefits:

- Prefill and decode can scale **independently** (e.g. more decode GPUs for long outputs,
  fewer prefill GPUs for short prompts).  
- You can pick **different GPU types** or precision configs for each phase.  
- Overall GPU utilization is higher under realistic, bursty traffic.

### NVIDIA “Dynamo”‑style distributed engines

NVIDIA’s newer serving stacks (often referred to as **Dynamo / distributed TensorRT‑LLM
plans**) push this idea further by producing a **cluster‑wide execution plan**:

- **Graph‑level partitioning**:
  - The compiled engine can be split across **tensor‑parallel, pipeline‑parallel, and
    even role‑specialized** nodes (prefill‑heavy vs decode‑heavy).  
  - The plan encodes where each layer or phase should run.
- **Runtime routing**:
  - A central controller decides which node runs prefill for a request, which nodes hold
    its KV cache, and how to shard the model when it does not fit on a single GPU.  
  - KV pages and activations can move between nodes according to the plan.
- **Tight integration with CUDA Graphs and quantization**:
  - Prefill graphs are captured per‑shard and replayed across the cluster.  
  - Decode uses paged KV, quantized KV, and continuous batching similar to vLLM, but at a
    **multi‑node scale**.

Conceptually, you can view these systems as:

- Layer 4–5 (**Graph + Runtime**) expanded from “inside one server” to **across many
  machines**, while Layer 6 (**Serving**) becomes responsible for routing requests between
  specialized pools (prefill vs decode, or different shards) based on the compiled plan.

## Glossary

- **Arithmetic Intensity (AI)**  
  FLOPs performed per byte read from HBM. High AI → likely compute‑bound; low AI → likely memory‑bound.

- **Roofline model**  
  Visual model that plots achieved FLOPs/s vs AI with two ceilings: one from compute peak, one from memory bandwidth. Shows whether a kernel is memory‑bound, compute‑bound, or launch‑limited.

- **Prefill**  
  First phase of inference that processes the entire prompt in parallel, builds initial KV cache, and is usually compute‑bound.

- **Decode**  
  Autoregressive generation phase that produces one (or a few) tokens per step, reusing cached K/V and typically memory‑bound.

- **KV cache**  
  Storage of attention keys and values for all processed tokens in each layer, so later tokens can attend to the past without recomputing everything.

- **Paged KV cache**  
  KV cache implementation that slices memory into fixed‑size pages and lets each sequence own a list of page pointers, reducing fragmentation and enabling efficient eviction/reuse.

- **Continuous batching**  
  Runtime strategy where the engine keeps a pool of active sequences and dynamically reshapes the decode batch every step, so new requests can join and finished ones leave without stopping the loop.

- **CUDA Graphs**  
  CUDA feature that records an entire computation as a graph and replays it with a single launch, removing most per‑kernel CPU overhead (especially useful for prefill).

- **FlashAttention**  
  Fused, tiled attention kernel that computes the same attention math as standard softmax attention but never materializes the full T×T score matrix in HBM, greatly reducing memory traffic.

- **TTFT (Time To First Token)**  
  Time from request arrival until the first token is emitted; dominated by prefill and CPU/graph overhead.

- **TPOT (Time Per Output Token)**  
  Average time between output tokens during steady‑state decode; dominated by memory‑bound attention + KV reads.

- **LoRA / QLoRA**  
  LoRA: low‑rank adapter fine‑tuning that adds small `A Bᵀ` updates on top of a frozen base model. QLoRA: combines a 4‑bit quantized base model with higher‑precision LoRA adapters so many fine‑tunes share one compressed backbone.

## One‑Screen Summary

- **Layer 1–2 (Hardware + Kernels)**: decide **how fast math can run** → AI, memory hierarchy, fusion, Tensor Cores.  
- **Layer 3–4 (Model + Graph)**: decide **what math to run and how to fuse it** → Transformer structure, attention variants, compilation, LoRA/QLoRA adapters.  
- **Layer 5–7 (Runtime + Serving + Profiling)**: decide **when and where to run it** → batching, KV cache, quantization, routing, adapter loading, and metrics.

The **inference stack** is the composition of all seven: change something at model level (e.g. GQA, FP8 KV, LoRA adapters) and it ripples down into **kernels, memory traffic, serving behavior, and multi‑tenant economics**.

