# Three-Tier LLM Inference with llama.cpp on KServe

This document describes how to deploy a GGUF language model on Kubernetes using KServe and llama.cpp with three-tier memory offloading: GPU VRAM → CPU RAM → NFS disk (via mmap). It covers how llama.cpp works internally, what each parameter in the deployment YAML does, how to place your model correctly, and includes real benchmark results from a working deployment.

---

## Table of Contents

1. [What is llama.cpp?](#what-is-llamacpp)
2. [How Three-Tier Memory Offloading Works](#how-three-tier-memory-offloading-works)
3. [This Deployment: Qwen2.5-0.5B Layer Split](#this-deployment-qwen25-05b-layer-split)
4. [YAML Reference: Every Parameter Explained](#yaml-reference-every-parameter-explained)
5. [Where to Put Your Model](#where-to-put-your-model)
6. [Deploying](#deploying)
7. [Testing the Endpoint](#testing-the-endpoint)
8. [Benchmark Results](#benchmark-results)
9. [Changing the Model or Layer Split](#changing-the-model-or-layer-split)

---

## What is llama.cpp?

llama.cpp is a C++ inference engine for large language models stored in the GGUF format. It was written to run LLMs efficiently on consumer hardware, including machines that do not have enough GPU VRAM to hold an entire model.

Key properties that make it useful here:

- **GGUF format**: a single binary file that stores model weights, tokeniser, and metadata together. No Python dependencies, no model sharding across multiple files.
- **Quantisation**: weights can be stored at reduced precision (Q4, Q5, Q8, etc.) to shrink the file. Q8_0 keeps 8 bits per weight — close to full precision but roughly half the size of float16.
- **Layer-level GPU offloading**: the model is made of stacked transformer layers. llama.cpp can put any subset of those layers on the GPU and run the rest on CPU. You choose the boundary with a single flag.
- **mmap**: llama.cpp memory-maps the GGUF file by default. The OS loads weight pages from disk only when they are first accessed, and can evict them under memory pressure. This means the model does not need to fully fit in RAM.
- **OpenAI-compatible HTTP server**: `llama-server` exposes `/v1/chat/completions`, `/v1/models`, `/health`, and other endpoints, so existing OpenAI client code works without modification.

---

## How Three-Tier Memory Offloading Works

```
┌─────────────────────────────────────────────────────┐
│  NFS / Disk  (tier 3 — slowest, largest)            │
│  The GGUF file lives here. mmap makes it look like  │
│  memory. Pages are faulted in on demand and evicted │
│  when RAM is tight. Cost: NFS latency on cold read. │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  CPU RAM  (tier 2 — medium speed and size)    │  │
│  │  Layers assigned to CPU live here. Once a     │  │
│  │  page is faulted in from NFS it stays in RAM  │  │
│  │  until the kernel reclaims it. The KV cache   │  │
│  │  for CPU layers is also here.                 │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  GPU VRAM  (tier 1 — fastest, smallest) │  │  │
│  │  │  Layers assigned to CUDA live here,     │  │  │
│  │  │  copied once at startup. Inference on   │  │  │
│  │  │  these layers runs at full GPU speed.   │  │  │
│  │  │  KV cache for GPU layers is also here.  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

During a forward pass llama.cpp processes all layers in order. Layers on the GPU run fast; layers on the CPU run slower; weights that have been evicted from RAM are faulted back in from the NFS mmap before they can be used. The model still produces correct output at every tier boundary — it just runs at the speed of its slowest active tier.

### Why mmap is the disk tier, not a separate cache

mmap does not copy file data into a separate buffer. It maps the file's pages directly into the process's virtual address space. When a weight tensor is needed, the CPU issues a page fault, the kernel reads the page from NFS into a physical RAM frame, and maps it in. If RAM fills up the kernel can evict that frame (writing nothing back because the file is read-only), and re-fault it later. From llama.cpp's perspective the tensor is always "in memory" — the OS handles the movement between disk and RAM transparently.

---

## This Deployment: Qwen2.5-0.5B Layer Split

Model: `qwen2.5-0.5b-instruct-q8_0.gguf`
Total layers: 24 transformer layers + 1 output layer = 25 total
`--n-gpu-layers 10` sends the last 10 layers plus the output layer to the GPU.

### Confirmed layer assignment from startup logs

```
layer  0 → CPU      layer  8 → CPU      layer 16 → CUDA0
layer  1 → CPU      layer  9 → CPU      layer 17 → CUDA0
layer  2 → CPU      layer 10 → CPU      layer 18 → CUDA0
layer  3 → CPU      layer 11 → CPU      layer 19 → CUDA0
layer  4 → CPU      layer 12 → CPU      layer 20 → CUDA0
layer  5 → CPU      layer 13 → CPU      layer 21 → CUDA0
layer  6 → CPU      layer 14 → CPU      layer 22 → CUDA0
layer  7 → CPU      layer 15 → CUDA0    layer 23 → CUDA0
                                        layer 24 → CUDA0 (output)
```

15 layers on CPU, 10 layers on GPU.

### Memory usage per tier

| Tier | What lives there | Measured usage |
|---|---|---|
| VRAM (GPU) | 10 transformer layers + output layer + GPU KV cache | ~612 MiB of 12288 MiB |
| RAM (CPU) | 15 transformer layers + CPU KV cache | ~1055 MiB |
| Disk (NFS mmap) | Full GGUF file, pages faulted in on demand | 512 MiB mapped (full file: ~638 MiB) |

KV cache split: 15 layers on CPU, 10 layers on CUDA0, total 24 MiB at 2048 context (float16).

---

## YAML Reference: Every Parameter Explained

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
```

Tells Kubernetes this is a KServe InferenceService object. KServe watches for these and creates the necessary pods, services, and routing.

```yaml
metadata:
  name: qwen-multitier-offload
```

The name of this InferenceService. It becomes part of pod names and service names. Change this when deploying a different model so deployments do not collide.

```yaml
  namespace: llm-demo
```

The Kubernetes namespace to deploy into. Must already exist. The PVC referenced in volumes must be in the same namespace.

```yaml
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
```

By default KServe uses Knative serverless mode with scale-to-zero. `RawDeployment` creates a plain Kubernetes Deployment instead — always running, no cold-start delays. Use this when you need predictable latency or when Knative is not installed.

```yaml
        image: ghcr.io/ggml-org/llama.cpp:server-cuda
```

The official llama.cpp server image with CUDA support. The `server-cuda` tag includes `llama-server` compiled against CUDA libraries. Alternative tags: `server` (CPU only), `server-rocm` (AMD GPU). Pin to a specific version tag in production (e.g. `server-cuda-b10820`) to avoid unexpected updates.

```yaml
          - "--model"
          - "/mnt/models/qwen2.5-0.5b-instruct-q8_0.gguf"
```

Path to the GGUF file inside the container. Must match the volume mountPath plus the filename on the NFS share. Update this when changing models.

```yaml
          - "--n-gpu-layers"
          - "10"
```

Number of transformer layers to offload to the GPU. llama.cpp assigns layers from the end — the last N layers go to GPU, the rest stay on CPU. For Qwen2.5-0.5B with 24 layers, `10` puts layers 15-24 on GPU and 0-14 on CPU. Set to `0` for CPU-only. Set higher than the total layer count to put everything on GPU. More GPU layers = faster inference but more VRAM consumed.

```yaml
          - "--ctx-size"
          - "2048"
```

Maximum context length in tokens — how much text (prompt + response combined) the model processes per call. Larger values increase KV cache memory on both CPU and GPU linearly. At 2048 tokens the KV cache is 24 MiB; at 32768 it would be ~384 MiB. Keep this as small as your use case allows.

```yaml
          - "--host"
          - "0.0.0.0"
```

IP address llama-server binds its HTTP listener to. `0.0.0.0` means all interfaces — required inside a container so Kubernetes can route traffic to it. Do not use `127.0.0.1` as no external traffic would reach the server.

```yaml
          - "--port"
          - "8080"
```

HTTP port llama-server listens on. Must match `containerPort` below. Note: llama.cpp's default port is changing to 9931 in a future release; pinning to 8080 here keeps the KServe health probes working regardless.

```yaml
          - "-v"
```

Enables verbose logging. Without this flag the layer assignment details are hidden. With `-v` the startup logs show which layers went to GPU vs CPU, KV cache sizes, and whether mmap is active — essential for verifying the three-tier split. Remove in production if log volume becomes an issue.

```yaml
        ports:
          - containerPort: 8080
            protocol: TCP
```

Tells Kubernetes which port the container exposes. Must match `--port` above. KServe uses this to create the Service that routes traffic to the pod.

```yaml
        resources:
          limits:
            cpu: '2'
            memory: "2Gi"
            nvidia.com/gpu: "1"
          requests:
            cpu: '500m'
            memory: "1Gi"
            nvidia.com/gpu: "1"
```

Kubernetes resource requests and limits:

- `cpu`: limit of 2 cores, request of 0.5 cores. llama.cpp uses multiple CPU threads for inference on CPU layers; more cores help but the bottleneck is usually memory bandwidth.
- `memory`: the container's RAM budget. Must cover the CUDA host context (~300-500 MiB), compute buffers, and the CPU KV cache. Do not set below 1.5 Gi for any CUDA process. The mmap'd weight pages are also charged to this cgroup but are reclaimable; anonymous memory is not.
- `nvidia.com/gpu`: requests one physical GPU. Must appear in both `limits` and `requests`. HAMI intercepts this request to schedule the pod onto a node with a free GPU. Without this the container starts without CUDA and `--n-gpu-layers` has no effect.

```yaml
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          periodSeconds: 10
          failureThreshold: 30
```

Probes `/health` every 10 seconds during startup, allowing up to 30 failures (300 seconds total). This is necessary because loading a large GGUF over NFS is slow — a ~638 MiB file can take 30-120 seconds. Without the extended startup probe the default readiness probe would kill the pod before llama-server finishes loading. The startup probe disables the readiness probe until it first succeeds.

```yaml
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          periodSeconds: 10
```

Once the startup probe passes, Kubernetes polls `/health` every 10 seconds. If it fails 3 consecutive times (default `failureThreshold`) the pod is marked NotReady and removed from the Service's endpoints. llama-server returns HTTP 200 when ready and HTTP 503 during loading.

```yaml
        volumeMounts:
          - name: model-storage
            mountPath: /mnt/models
```

Mounts the volume named `model-storage` at `/mnt/models` inside the container. The `--model` argument path must start with this mountPath. The name must match the entry in `volumes` below.

```yaml
    volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: qwen-gguf-pvc
```

Declares the volume as a PersistentVolumeClaim. The PVC `qwen-gguf-pvc` must exist in namespace `llm-demo` and be bound to a PersistentVolume containing the GGUF file. In this deployment the PVC is backed by NFS, providing the disk tier.

---

## Where to Put Your Model

The model file must be accessible via a PersistentVolumeClaim in the same namespace as the InferenceService.

### Directory structure on the NFS server

```
/mnt/nfs/models/
└── llm-demo-qwen-gguf-pvc-pvc-<uuid>/
    └── qwen2.5-0.5b-instruct-q8_0.gguf
```

The subdirectory is created automatically by the NFS provisioner when the PVC is bound. Copy your GGUF file into it:

```bash
# Find the provisioned directory
ls /mnt/nfs/models/

# Copy your model in
cp /path/to/your-model.gguf /mnt/nfs/models/llm-demo-qwen-gguf-pvc-pvc-<uuid>/
```

### NFS export requirements

The NFS export on the server must include the `insecure` option to allow mounts from worker nodes, which use unprivileged source ports (>1024):

```
/mnt/nfs/models *(rw,sync,no_subtree_check,no_root_squash,insecure)
```

Without `insecure`, worker nodes get `Operation not permitted` when kubelet tries to mount the volume. After editing `/etc/exports` reload with:

```bash
sudo exportfs -ra
sudo exportfs -v  # confirm insecure is shown
```

### Using a local path instead of NFS

If the pod is pinned to a specific node, you can use a hostPath volume:

```yaml
    volumes:
      - name: model-storage
        hostPath:
          path: /data/models
          type: Directory
    nodeSelector:
      kubernetes.io/hostname: your-node-name
```

Add the `nodeSelector` under `spec.predictor` alongside `containers`.

---

## Deploying

```bash
# Create the namespace if it doesn't exist
kubectl create namespace llm-demo

# Apply the InferenceService
kubectl apply -f llama-3-fixed.yaml

# Watch the pod come up (takes 1-5 minutes due to NFS model loading)
kubectl get pod -n llm-demo -w

# Check the InferenceService is Ready
kubectl get inferenceservice qwen-multitier-offload -n llm-demo
```

### Verify the three-tier split after startup

```bash
POD=$(kubectl get pod -n llm-demo \
  -l serving.kserve.io/inferenceservice=qwen-multitier-offload \
  -o name | head -1)

# Layer assignment
kubectl logs -n llm-demo $POD | grep -E "assigned to device|offloaded"

# mmap confirmation
kubectl logs -n llm-demo $POD | grep mmap

# VRAM usage — run on the worker node
nvidia-smi
```

Expected output:

```
layer  0 assigned to device CPU
...
layer 15 assigned to device CUDA0
...
offloaded 10/25 layers to GPU
load_mode = mmap
```

---

## Testing the Endpoint

```bash
# Port-forward to the pod
kubectl port-forward -n llm-demo $POD 8080:8080

# Health check
curl http://localhost:8080/health

# List models (shows the exact model name string to use in requests)
curl http://localhost:8080/v1/models | python3 -m json.tool

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/models/qwen2.5-0.5b-instruct-q8_0.gguf",
    "messages": [{"role": "user", "content": "Hello, what are you?"}],
    "max_tokens": 100
  }'
```

The `model` field in the request must match the full path string the server reports. Check `GET /v1/models` to confirm the exact value.

---

## Benchmark Results

**Hardware**: NVIDIA RTX 3080 Ti (12 GB VRAM) on worker node `new2`
**Model**: Qwen2.5-0.5B-Instruct Q8_0 (~638 MiB)
**Split**: 10/25 layers on GPU, 15 layers on CPU, weights on NFS via mmap
**Context size**: 2048 tokens

### Single request

| Metric | Value |
|---|---|
| Output tokens generated | 102 |
| Generation speed | 15.07 tokens/sec |
| Time to first token (TTFT) | 224 ms |
| Total wall-clock time | 7.0 s |

### 4 concurrent requests

| Metric | Value |
|---|---|
| Per-slot generation speed | 6.62 tokens/sec |
| Combined throughput | ~26.5 tokens/sec |

All 4 slots ran at the same per-slot speed. llama.cpp batches all active slots into each forward pass — compute is shared, not serialised. The total throughput scales close to linearly with the number of concurrent slots up to the hardware limit.

### Prompt cache effect

| Run | Prompt eval time | Generation speed |
|---|---|---|
| First (cold cache) | 279 ms | 14.76 tokens/sec |
| Second (same prompt, cache hit) | 88 ms | 15.47 tokens/sec |

The second request reused the cached KV state, reducing prompt eval time by 68%. Generation speed is nearly identical since that phase is compute-bound.

### What the numbers mean

15 tokens/sec on a mixed CPU/GPU split is expected for a 0.5B Q8 model. The GPU layers run fast; the CPU layers are the bottleneck. Moving more layers to GPU increases throughput proportionally until VRAM is exhausted. Putting all 25 layers on this 12 GB card is feasible (~620 MiB total VRAM) and would push throughput toward 80-100 tokens/sec.

---

## Changing the Model or Layer Split

### Put more layers on the GPU

```yaml
- "--n-gpu-layers"
- "24"   # all transformer layers; output layer offloads automatically
```

Check available VRAM first:

```bash
# On the worker node:
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

### Run CPU-only (no GPU)

```yaml
- "--n-gpu-layers"
- "0"
```

Remove GPU resource lines from both `limits` and `requests`.

### Increase context size

```yaml
- "--ctx-size"
- "8192"
```

Also increase the memory limit — KV cache grows linearly with context size.

### Approximate VRAM per layer at Q8 quantisation

| Model size | Approx VRAM per layer |
|---|---|
| 0.5B | ~25 MiB |
| 7B | ~130 MiB |
| 13B | ~240 MiB |
| 70B | ~1200 MiB |

On a 12 GB card, a 7B Q8 model can fit roughly 8-9 layers on GPU — enough to meaningfully accelerate inference while keeping the rest on CPU and NFS.
