# KV Cache Optimization for Small Models on Multi-GPU vLLM Deployments

*Based on deployment of Qwen2.5-0.5B-Instruct via KServe LLMInferenceService with tensor parallelism across 2x RTX 3080 Ti (12GB each)*

---

## What is KV Cache and Why Does It Matter

During inference, a transformer model computes Key and Value matrices for every token in the context. These are stored in GPU memory so they don't need to be recomputed on each new token generation. This storage is the KV cache.

For a small model like Qwen2.5-0.5B-Instruct, the model weights themselves are tiny (~1GB). The problem is that vLLM pre-allocates KV cache to fill whatever GPU memory remains after the model loads — which on a 12GB card is roughly 10GB+ of KV cache for a single sequence slot, and it tries to warm up with 256 dummy requests simultaneously, multiplying that demand many times over.

This is why a 0.5B parameter model — which should comfortably fit in a few hundred MB — can OOM a 12GB GPU.

---

## What We Ran Into

### Deployment: Single GPU (Working Baseline)

```
nvidia.com/gpu: "1"
```

The model loaded fine and used ~11.5GB of VRAM on GPU 1 alone. GPU 0 was idle. This worked because vLLM defaulted to `gpu_memory_utilization=0.9`, pre-allocating 90% of the available 12GB for KV cache. With only one GPU and no warmup pressure, it stayed just under the limit.

### Deployment: 2 GPU Tensor Parallel (First Attempt — OOM)

```
nvidia.com/gpu: "2"
VLLM_ADDITIONAL_ARGS: "--tensor-parallel-size 2"
```

Both GPUs loaded `Worker_TP0` and `Worker_TP1` with ~10.6GB each, then crashed during the warmup phase:

```
CUDA out of memory. Tried to allocate 446.00 MiB.
GPU 1 has a total capacity of 11.63 GiB of which 1.16 GiB is free.
CUDA out of memory occurred when warming up sampler with 256 dummy requests.
```

The root cause: vLLM with default settings tried to warm up 256 concurrent sequences. Each sequence needs its own KV cache slot. With `max_model_len=32768` (the model's default), the KV cache per sequence is enormous, and 256 of them at once exhausted both GPUs simultaneously.

The HAMI scheduler also left its own footprint (`cuMemoryAllocate failed res=2`) indicating that HAMi's virtual GPU allocator was interfering with PyTorch's CUDA memory allocator during the warmup phase.

### Deployment: 2 GPU Tensor Parallel (Fixed)

```
VLLM_ADDITIONAL_ARGS: "--tensor-parallel-size 2 --gpu-memory-utilization 0.75 --max-num-seqs 32 --max-model-len 8192"
```

This worked. Both GPUs stabilized at ~9.7GB each, leaving ~2.5GB free per GPU for warmup overhead, sampler operations, and NCCL all-reduce buffers.

---

## The Three Levers

### 1. `--gpu-memory-utilization`

Controls what fraction of available VRAM vLLM pre-allocates for KV cache after loading model weights.

| Value | VRAM reserved (per 12GB GPU) | Free headroom |
|---|---|---|
| 0.90 (default) | ~10.7GB | ~1.3GB |
| 0.75 (used) | ~8.9GB | ~3.1GB |
| 0.60 | ~7.1GB | ~4.9GB |

Lower values leave more room for warmup, NCCL buffers, and sampler operations. For small models where the weights don't fill the GPU, the default 0.90 is almost always too aggressive.

### 2. `--max-num-seqs`

Controls the maximum number of sequences vLLM will process concurrently. This directly controls warmup memory pressure.

- Default: 256
- Used: 32

vLLM pre-allocates KV cache for `max_num_seqs` sequences simultaneously during warmup. Reducing this from 256 to 32 cuts the warmup memory requirement by 8x. For a small model serving modest traffic, 32 concurrent sequences is more than sufficient.

### 3. `--max-model-len`

Controls the maximum context length in tokens. The KV cache size scales linearly with this value.

- Default for Qwen2.5-0.5B: 32768 tokens
- Used: 8192 tokens

Cutting from 32768 to 8192 reduces KV cache size per sequence by 4x. For most practical use cases (chat, Q&A, summarization of short documents), 8192 tokens is more than enough. You only need 32k if you're processing very long documents.

---

## How These Interact

The total KV cache memory requirement during warmup is approximately:

```
KV cache memory ≈ max_num_seqs × max_model_len × num_layers × hidden_size × dtype_bytes
```

For Qwen2.5-0.5B (24 layers, hidden size 896, bfloat16):

| Config | Approx warmup KV memory |
|---|---|
| Default (256 seqs, 32768 len) | ~45GB per GPU |
| Fixed (32 seqs, 8192 len) | ~1.4GB per GPU |

This is why the defaults are completely wrong for small models on consumer GPUs — they were tuned for large models on datacenter A100/H100 cards with 40-80GB VRAM where even the default warmup fits comfortably.

---

## Tensor Parallelism and KV Cache

With `--tensor-parallel-size 2`, the model weights are sharded across both GPUs. The KV cache however is NOT shared — each GPU maintains its own KV cache for its shard of the attention heads.

This means:
- Model weights: split in half across GPUs ✅
- KV cache: each GPU holds its own copy for its attention heads ✅
- NCCL all-reduce: synchronizes activations between GPUs on every forward pass ✅

The practical implication is that tensor parallelism does not reduce KV cache pressure per GPU. It reduces the memory needed for weights, but the KV cache budget on each GPU remains the same. On 2x 12GB GPUs you still need to fit the KV cache within ~12GB per card (minus weights, minus headroom).

---

## What to Tackle Next

### Chunked Prefill (Already Enabled)

vLLM v0.15.1 had chunked prefill enabled by default (`enable_chunked_prefill=True` visible in the logs). This helps throughput by batching prefill and decode phases together but does not directly reduce peak KV cache memory.

### Prefix Caching (Already Enabled)

Prefix caching (`enable_prefix_caching=True`) was also on by default. This reuses KV cache across requests that share a common prefix (e.g. a system prompt). Highly effective if your deployment uses a fixed system prompt — the KV cache for that prefix is computed once and reused across all requests, reducing effective cache pressure for repeated patterns.

### Quantized KV Cache

vLLM supports `--kv-cache-dtype fp8` which halves the KV cache memory footprint by storing keys and values in 8-bit floats instead of bfloat16. This would allow either doubling `max_num_seqs`, doubling `max_model_len`, or reclaiming ~4GB per GPU. RTX 3080 Ti supports FP8 operations.

```
VLLM_ADDITIONAL_ARGS: "--tensor-parallel-size 2 --gpu-memory-utilization 0.75 --max-num-seqs 32 --max-model-len 8192 --kv-cache-dtype fp8"
```

### Sliding Window Attention

Some models support sliding window attention, where only the last N tokens are kept in the KV cache rather than the full context. Qwen2.5 does not use this by default but it can be configured at the model architecture level for future model choices.

### Pipeline Parallelism vs Tensor Parallelism

For very deep models, pipeline parallelism (`--pipeline-parallel-size`) splits layers across GPUs rather than splitting each layer. This reduces NCCL communication overhead but adds pipeline bubbles. For Qwen2.5-0.5B (24 layers) on 2 GPUs, tensor parallelism is the right choice. Pipeline parallelism becomes relevant at larger model sizes or more GPUs.

---

## Reference: Working Configuration

```yaml
env:
  - name: HF_HOME
    value: "/mnt/models"
  - name: VLLM_ADDITIONAL_ARGS
    value: "--tensor-parallel-size 2 --gpu-memory-utilization 0.75 --max-num-seqs 32 --max-model-len 8192"
resources:
  limits:
    nvidia.com/gpu: "2"
    memory: 12Gi
    cpu: "2"
  requests:
    nvidia.com/gpu: "2"
    memory: 4Gi
    cpu: "100m"
```

**Resulting VRAM usage:** ~9.7GB per GPU  
**Available KV cache:** ~7.99GB per GPU  
**Max concurrent sequences:** 32  
**Max context length:** 8192 tokens  
**GPU-Util during inference:** 58% (TP0) / 99% (TP1)
