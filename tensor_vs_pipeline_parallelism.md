# Tensor Parallelism vs Pipeline Parallelism
*Practical guide based on Qwen2.5-0.5B-Instruct deployment on 2x RTX 3080 Ti (12GB each) via KServe LLMInferenceService*

---

## The Core Idea

Both strategies split a model across multiple GPUs, but they cut it differently.

**Tensor Parallelism (TP)** cuts each layer *horizontally* — every GPU works on every layer at the same time, each handling a slice of it. They must communicate after every single layer to combine results.

**Pipeline Parallelism (PP)** cuts the model *vertically* — GPU 0 handles the first half of layers, passes its output to GPU 1 which handles the second half. They work one after the other, not simultaneously.

```
Tensor Parallelism (TP=2):
┌─────────────────────────────────────────┐
│ Layer 1   [GPU0: heads 1-12 | GPU1: heads 13-24] → sync
│ Layer 2   [GPU0: heads 1-12 | GPU1: heads 13-24] → sync
│ ...
│ Layer 24  [GPU0: heads 1-12 | GPU1: heads 13-24] → sync
└─────────────────────────────────────────┘

Pipeline Parallelism (PP=2):
┌─────────────────────────────────────────┐
│ Layers 1-12   → GPU0 runs all of them
│                      ↓ pass activations
│ Layers 13-24  → GPU1 runs all of them
└─────────────────────────────────────────┘
```

---

## How They Differ on Memory

This is the most important practical difference.

### Tensor Parallelism — weights split, KV cache NOT split

Each GPU holds half the weight matrices (each attention layer's Q/K/V/O projections are split by attention head). However, each GPU still needs to maintain KV cache for ALL tokens in the context for its own heads. The KV cache budget per GPU is essentially unchanged.

For Qwen2.5-0.5B on 2x 12GB GPUs:
- Model weights per GPU: ~0.23GB (half of ~0.46GB total)
- KV cache per GPU: ~9.4GB (same as single GPU, just for half the heads)
- **Result: ~9.7GB per GPU**

### Pipeline Parallelism — weights split AND KV cache split

GPU 0 runs layers 1-12 and stores KV cache only for those layers. GPU 1 runs layers 13-24 and stores KV cache only for those layers. Since KV cache size scales with the number of layers, each GPU needs roughly half.

For Qwen2.5-0.5B on 2x 12GB GPUs:
- Model weights per GPU: ~0.23GB (half of ~0.46GB total)
- KV cache per GPU: ~4.7GB (half the layers = half the KV cache)
- **Result: ~5-6GB per GPU**

---

## Communication Pattern

### Tensor Parallelism
After every single transformer layer, all GPUs must synchronize via an **all-reduce** operation over NCCL. For 24 layers, that's 24 synchronization points per forward pass. Each sync is small (just the layer output activations) but frequent.

This requires high GPU-to-GPU bandwidth. On NVLink (datacenter GPUs) this is fast. On PCIe (consumer GPUs like RTX 3080 Ti) this is a bottleneck.

### Pipeline Parallelism
GPU 0 processes its layers and passes the full activation tensor to GPU 1 exactly once per forward pass. GPU 1 does the same in reverse during backpropagation (training) or just returns the output (inference). Only 1 synchronization point, but the payload is larger.

This works fine over PCIe and requires far less interconnect bandwidth.

---

## The Pipeline Bubble

Pipeline parallelism has one inherent inefficiency: when GPU 0 is computing, GPU 1 is idle waiting. When GPU 1 is computing, GPU 0 is idle. This is called the pipeline bubble.

```
Time →
GPU0: [computing layers 1-12]  [idle]                [computing layers 1-12]
GPU1: [idle]                   [computing layers 13-24] [idle]
```

For inference (not training), the bubble is small because the forward pass is short. For a 0.5B model with 24 layers split 12/12, each GPU's compute time is roughly equal, so GPU 1 starts almost immediately after GPU 0 finishes. The latency overhead is modest.

---

## When to Use Each

| Situation | Use |
|---|---|
| Single layer too large for one GPU (e.g. 70B model) | Tensor Parallel |
| Model fits on one GPU but you want to spread the load | Pipeline Parallel |
| You have NVLink between GPUs | Tensor Parallel |
| You have only PCIe between GPUs (consumer cards) | Pipeline Parallel |
| You need lowest possible latency per token | Tensor Parallel |
| You want to minimize VRAM per GPU | Pipeline Parallel |
| Many layers, moderate hidden size | Pipeline Parallel |
| Few very wide layers (large hidden dim) | Tensor Parallel |
| Small model like Qwen2.5-0.5B on RTX 3080 Ti | **Pipeline Parallel** |
| Large model like Llama-70B on A100s | **Tensor Parallel** |

---

## Observed Behavior During Inference

### Tensor Parallel (what we saw)
Both GPUs spike in GPU-Util simultaneously because they are computing the same layer in parallel and syncing after each one.

```
nvidia-smi during inference (TP):
GPU 0: 58%  util, 135W, 9716MiB
GPU 1: 99%  util, 165W, 9716MiB
← Both active at the same time
```

GPU 1 runs slightly hotter because rank 1 handles the final all-reduce and sampling step on top of its share of the layer compute.

### Pipeline Parallel (expected)
GPU 0 spikes first while computing layers 1-12, then GPU 1 spikes while computing layers 13-24. They alternate rather than running simultaneously.

```
nvidia-smi during inference (PP):
GPU 0: 85%  util → drops → 0%  (after handing off to GPU 1)
GPU 1: 0%   util → spikes → 80% (after receiving from GPU 0)
← Sequential activity
```

VRAM per GPU should drop to ~5-6GB compared to ~9.7GB with TP.

---

## Practical Examples

### Example 1 — Tensor Parallelism (TP=2)

Splits each attention layer across both GPUs. Both GPUs active simultaneously. Higher VRAM per GPU but lower latency. Used when individual layers are too large for one GPU or when NVLink is available.

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: qwen2-0b5-instruct-local
  namespace: llm-demo
spec:
  model:
    name: Qwen/Qwen2.5-0.5B-Instruct
    uri: pvc://qwen-local-pv-claim/
  replicas: 1
  router:
    scheduler: {}
    route: {}
    gateway:
      refs:
        - name: kserve-ingress-gateway
          namespace: kserve
  template:
    containers:
      - name: main
        image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
        resources:
          limits:
            cpu: '2'
            memory: 12Gi
            nvidia.com/gpu: "2"
          requests:
            cpu: '100m'
            memory: 4Gi
            nvidia.com/gpu: "2"
        env:
          - name: HF_HOME
            value: "/mnt/models"
          - name: VLLM_ADDITIONAL_ARGS
            value: "--tensor-parallel-size 2 --gpu-memory-utilization 0.75 --max-num-seqs 32 --max-model-len 8192"
```

**What you get:**
- VRAM per GPU: ~9.7GB
- KV cache per GPU: ~7.99GB
- Both GPUs active simultaneously during inference
- 24 NCCL all-reduce syncs per forward pass over PCIe
- Good fit for: large models, NVLink setups, latency-critical workloads

---

### Example 2 — Pipeline Parallelism (PP=2)

Splits the model by layers — GPU 0 runs layers 1-12, GPU 1 runs layers 13-24. KV cache is also split by layer so VRAM per GPU is roughly halved. Better fit for consumer GPUs connected over PCIe.

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: qwen2-0b5-instruct-local
  namespace: llm-demo
spec:
  model:
    name: Qwen/Qwen2.5-0.5B-Instruct
    uri: pvc://qwen-local-pv-claim/
  replicas: 1
  router:
    scheduler: {}
    route: {}
    gateway:
      refs:
        - name: kserve-ingress-gateway
          namespace: kserve
  template:
    containers:
      - name: main
        image: 127.0.0.1:5000/kserve/huggingfaceserver:v0.17.0-gpu
        resources:
          limits:
            cpu: '2'
            memory: 12Gi
            nvidia.com/gpu: "2"
          requests:
            cpu: '100m'
            memory: 4Gi
            nvidia.com/gpu: "2"
        env:
          - name: HF_HOME
            value: "/mnt/models"
          - name: VLLM_ADDITIONAL_ARGS
            value: "--pipeline-parallel-size 2 --gpu-memory-utilization 0.75 --max-num-seqs 32 --max-model-len 8192"
```

**What you get:**
- VRAM per GPU: ~5-6GB (roughly half of TP)
- KV cache split across GPUs by layer
- GPUs active sequentially during inference (alternating spikes)
- Only 1 activation handoff between GPUs per forward pass
- Good fit for: small models, consumer PCIe GPUs, VRAM-constrained setups

---

## Side-by-Side Summary

| | Tensor Parallel (TP=2) | Pipeline Parallel (PP=2) |
|---|---|---|
| vLLM flag | `--tensor-parallel-size 2` | `--pipeline-parallel-size 2` |
| Splits | Each layer by attention head | Layers 1-12 vs 13-24 |
| KV cache per GPU | Full (not reduced) | Half (split by layer) |
| VRAM per GPU (our setup) | ~9.7GB | ~5-6GB |
| GPU activity pattern | Both simultaneously | Sequential / alternating |
| NCCL syncs per forward pass | 24 (one per layer) | 1 (one handoff) |
| Interconnect requirement | High bandwidth (NVLink ideal) | Low bandwidth (PCIe fine) |
| Per-token latency | Lower | Slightly higher |
| Best model size | Large (layers too big for 1 GPU) | Small-medium (fits on 1 GPU) |
| Best GPU setup | Datacenter (A100, H100, NVLink) | Consumer (RTX, PCIe) |
| Our hardware fit | Suboptimal (PCIe bottleneck) | Better fit |
