# Qwen2.5-0.5B on KServe with DeepSpeed ZeRO-Inference (NVMe/CPU param offload)

A working deployment of Qwen2.5-0.5B, served through KServe, running on DeepSpeed's **ZeRO-Inference**: parameters live in GPU VRAM, host RAM, and local disk, and stream into VRAM layer-by-layer right before each forward pass needs them, then get evicted again. This README documents what was actually built, why each piece is the way it is, and the exact steps to deploy it from scratch.

Cluster: `3080-1` (k3s control plane, no GPU) + `new2` (k3s worker, GPU via HAMi vGPU, shared with other cluster infrastructure).

---

## 1. What this actually is

DeepSpeed ZeRO stage 3 partitions a model's parameters instead of keeping the whole thing resident on GPU. For inference specifically ("ZeRO-Inference"), that partitioning engine keeps most parameters in CPU RAM or on local NVMe/disk, and streams each layer into GPU memory immediately before it's used in the forward pass, then evicts it. This is real dynamic residency management, not a static "these layers live here" assignment — DeepSpeed decides what's resident based on a live-parameter budget and prefetch distance, not a fixed per-layer map.

**The API that actually does this**: `transformers.deepspeed.HfDeepSpeedConfig` + `deepspeed.initialize()`, using a ZeRO-3 config with `offload_param`. This is distinct from `deepspeed.init_inference()`, which is a *different* DeepSpeed technology (tensor-parallel kernel injection) and does **not** do CPU/NVMe parameter streaming. Getting this API choice right matters — the two are easy to conflate and solve different problems.

**Why 0.5B specifically**: Qwen2.5-0.5B is ~1GB in fp16 and fits comfortably in any GPU's VRAM on its own. The point of this deployment was to validate the *mechanism* — correctness of the whole pipeline, not a performance test. The `ds_config.json` below deliberately sets aggressive live-parameter thresholds to force DeepSpeed to evict/reload parameters it wouldn't otherwise need to move, so the swap behavior is actually observable (watch disk I/O on the offload path during a request — see section 8).

---

## 2. Why the architecture is shaped the way it is

**Everything GPU/RAM/disk-related runs on one node (`new2`), not split across the cluster.** ZeRO-Inference's parameter swap hierarchy only works within a single process on a single machine — the GPU, the RAM it offloads to, and the NVMe/disk it offloads to must all be local to that one node. It cannot use `3080-1`'s disk as a "tier" while the GPU runs on `new2`. Since `3080-1` is short on disk space and `new2` has the GPU, the whole runtime — model load, GPU compute, CPU offload buffer, disk swap files — lives on `new2`, pinned there with a KServe `nodeSelector`.

**The model checkpoint lives on NFS; the live offload swap path does not.** The Qwen weights are read once at startup from `/mnt/nfs/...` (NFS-backed, fine for sequential reads). The *live* ZeRO-3 swap files — written and read continuously during inference — live on `new2`'s local root disk at `/mnt/ds-offload`, mounted via `hostPath`/PVC. `new2` turned out to have no dedicated NVMe partition; its only local (non-NFS) disk is `/dev/vda1`, with limited free space, which was enough for the 0.5B test but is a real constraint to revisit for larger models.

---

## 3. The real build issues hit, and what fixed each one

This deployment hit a sequence of real, specific failures — documenting them here because the fixes are non-obvious and will recur if this image or config is ever rebuilt from scratch.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `nvcr.io` 403 Forbidden pulling the CUDA devel base image | NVIDIA's NGC registry now requires auth even for some previously-anonymous public images | Switched base image to `docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04` (Docker Hub mirror, no auth needed) |
| 2 | Deciding whether to rebuild from a CUDA devel image at all | Already had a working 23.8GB `qwen-hf-runtime:v1` image with CUDA *runtime* libs but no compiler | Checked for `nvcc` in the existing image (`which nvcc` — not found), confirmed it's a runtime-only base; installed just `cuda-nvcc-12-1` + `cuda-cudart-dev-12-1` + `libaio-dev` via NVIDIA's apt repo directly into the existing image rather than pulling a whole separate devel image |
| 3 | `deepspeed==0.14.4` pip install failing: `fatal error: oneapi/ccl.hpp: No such file or directory` | DeepSpeed's ahead-of-time (AOT) op build tried to compile the CCL (Collective Communication Library) CPU comm backend, which needs Intel oneAPI headers that aren't installed and aren't needed for single-GPU inference | Added `ENV DS_BUILD_CCL_COMM=0` to the Dockerfile's op-disable list (alongside `SPARSE_ATTN`, `EVOFORMER_ATTN`, `TRANSFORMER`, etc. — DeepSpeed's AOT build aborts entirely if *any* enabled op fails to compile, so every unneeded op must be explicitly disabled) |
| 4 | Container starts, crashes: `pydantic.v1.error_wrappers.ValidationError: 1 validation error for DeepSpeedZeroConfig — aio: extra fields not permitted` | `aio` was nested inside `zero_optimization` in `ds_config.json`; DeepSpeed 0.14.4's schema expects `aio` as a **top-level** config key, not nested | Moved the `aio` block out of `zero_optimization` to be a sibling top-level key in `ds_config.json` |
| 5 | Standalone `docker run` hung for a long time after `AsyncPartitionedParameterSwapper` initialized | Not actually a bug — ZeRO-3 writes every parameter tensor's partition out to the swap folder as part of initial partitioning, even for a small model; this is slow on non-NVMe disk | Confirmed real progress via `iostat -x 1` (disk writes happening) and `du -sh /mnt/ds-offload` (directory growing) rather than assuming a hang |
| 6 | k3s pod stuck `ContainerCreating`: `failed to pull ... rancher/mirrored-pause:3.10.2 ... 403 Forbidden` | Same anonymous-pull restriction pattern as issue #1, this time for Kubernetes' internal sandbox/pause container from Docker Hub | Not fully resolved via registry login in this session — see open items |
| 7 | Pod `ImagePullBackOff`: `http: server gave HTTP response to HTTPS client` pulling from `10.42.0.1:5000` | The local registry is plain HTTP; `containerd` (via k3s) defaults to HTTPS and doesn't share Docker daemon's separate insecure-registry config | Added `/etc/rancher/k3s/registries.yaml` on `new2` declaring the registry as an HTTP mirror with `insecure_skip_verify: true`, then `systemctl restart k3s-agent` |
| 8 | `model.py` crash: `TabError: inconsistent use of tabs and spaces in indentation` | A later edit to `model.py` mixed tabs and spaces around the startup print statement — Python 3 refuses to run on this | `sed -i 's/\t/    /g' model.py`, verified with `python3 -m py_compile model.py`, rebuilt and pushed the image |
| 9 | Pod crash: `ModuleNotFoundError: No module named 'mpi4py'` inside `deepspeed.zero.Init()` → `init_distributed()` → `mpi_discovery()` | `deepspeed.initialize()` needs to know it's a single-process, non-distributed run; without the right env vars it falls back to attempting MPI-based peer discovery | Added `MASTER_ADDR=127.0.0.1`, `MASTER_PORT=29500`, `RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=1` to the InferenceService pod's `env:` block (these were already correct in the manual `docker run` test command, but had never been added to the Kubernetes manifest) |
| 10 | Pod `Pending`: `hami-scheduler ... Insufficient memory` | `new2` is a shared cluster node running ~48 pods of cluster infrastructure (Istio, cert-manager, Knative, KEDA, GPU operator, monitoring), already at 93% memory *requested* / 184% memory *limits* before this workload — the original `8Gi` request guide default didn't fit | Lowered `resources.requests.memory` to `1Gi` (from 8Gi) and `limits.memory` to `2Gi` (from 16Gi) — generous enough for a 0.5B model's offload buffers, small enough to fit the node's real headroom |
| 11 | Confusing duplicate pods/ReplicaSets during iteration | Each `isvc.yaml` edit + `kubectl apply` creates a *new* ReplicaSet (pod-template-hash changes); the old ReplicaSet isn't automatically cleaned up if its pods never reach `Ready` | Explicitly `kubectl delete rs <old-hash>` after confirming the new one is correct, every time the spec changes during debugging |

**Open/unresolved from this session**: issue #6 (pause-image pull 403) was circumvented rather than explicitly fixed with a documented root cause — if it recurs, check Docker Hub anonymous pull limits / login on `new2` specifically for the `rancher/mirrored-pause` image namespace.

---

## 4. Final file layout

```
qwen-deepspeed-runtime/
├── Dockerfile
├── model.py
├── ds_config.json
└── isvc.yaml           (applied separately via kubectl, not part of the image)
```

### `ds_config.json` (corrected — `aio` top-level)

```json
{
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local-offload",
      "pin_memory": true,
      "buffer_count": 4,
      "buffer_size": 1e8,
      "max_in_cpu": 5e8
    },
    "offload_optimizer": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e8,
    "stage3_max_reuse_distance": 1e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  },
  "train_batch_size": 1,
  "train_micro_batch_size_per_gpu": 1
}
```

### `model.py` — key points (full file was iterated on; this documents the load-bearing parts)

- `HfDeepSpeedConfig(ds_config)` is stored as `self.dschf` (not a local variable) — HuggingFace's own documentation states this object must remain alive for the process's lifetime, since it's held internally as a `weakref`. A local variable in `load()` would go out of scope as soon as `load()` returns, which is exactly wrong for a server process that keeps running after startup.
- `HfDeepSpeedConfig` is constructed **before** `AutoModelForCausalLM.from_pretrained(...)`, so HuggingFace loads the model with ZeRO-3 partitioning awareness from the start, rather than fully materializing it on one device first.
- `deepspeed.initialize()` is called directly (not via the `deepspeed` CLI launcher) — single-GPU ZeRO-Inference doesn't need multi-process orchestration, but does need the single-process distributed env vars present (see `isvc.yaml` below), or it falls back to MPI peer discovery and crashes.
- Indentation is consistent spaces throughout — verify with `python3 -m py_compile model.py` before any rebuild.

### Dockerfile — key points

```dockerfile
FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
# (or: build on top of an existing CUDA runtime image + apt-installed nvcc,
#  if reusing an existing image rather than starting fresh — see issue #2 above)

# libaio-dev required for DeepSpeed's NVMe async I/O offload engine
RUN apt-get update && apt-get install -y --no-install-recommends libaio-dev ...

# Explicitly disable every DeepSpeed op not needed for single-GPU ZeRO-Inference —
# DS_BUILD_OPS=1 aborts the entire install if ANY enabled op fails to compile
ENV DS_BUILD_OPS=1
ENV DS_BUILD_AIO=1
ENV DS_BUILD_CPU_ADAM=0
ENV DS_BUILD_FUSED_ADAM=0
ENV DS_BUILD_UTILS=1
ENV DS_BUILD_SPARSE_ATTN=0
ENV DS_BUILD_EVOFORMER_ATTN=0
ENV DS_BUILD_TRANSFORMER=0
ENV DS_BUILD_TRANSFORMER_INFERENCE=0
ENV DS_BUILD_STOCHASTIC_TRANSFORMER=0
ENV DS_BUILD_QUANTIZER=0
ENV DS_BUILD_CCL_COMM=0
RUN pip install --no-cache-dir deepspeed==0.14.4

# Verify with DeepSpeed's own tool — don't assume a non-error exit means
# async_io actually built; check the report explicitly
RUN ds_report 2>&1 | tee /tmp/ds_report.log && \
    if grep -qE "async_io.*\[NO\]" /tmp/ds_report.log; then \
        echo "FATAL: async_io op did not build"; exit 1; \
    fi
```

### `isvc.yaml` — final working shape

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen2-zero-inference
  namespace: llm-demo
spec:
  predictor:
    nodeSelector:
      workload-node: zero-inference
    containers:
      - name: kserve-container
        image: 10.42.0.1:5000/kserve/qwen-zero-inference:v1
        imagePullPolicy: Always   # avoids stale-local-image confusion during iteration
        args:
          - --http_port=8080
          - --model_name=qwen2-zero-inference
        env:
          - name: MODEL_DIR
            value: /mnt/models
          - name: DS_CONFIG_PATH
            value: /app/ds_config.json
          - name: MASTER_ADDR
            value: "127.0.0.1"
          - name: MASTER_PORT
            value: "29500"
          - name: RANK
            value: "0"
          - name: LOCAL_RANK
            value: "0"
          - name: WORLD_SIZE
            value: "1"
        resources:
          limits:
            cpu: "2"
            memory: 2Gi
            nvidia.com/gpu: "1"
          requests:
            cpu: "500m"
            memory: 1Gi
            nvidia.com/gpu: "1"
        volumeMounts:
          - name: model-storage
            mountPath: /mnt/models
            readOnly: true
          - name: offload-storage
            mountPath: /local-offload
    volumes:
      - name: model-storage
        hostPath:
          path: /mnt/nfs/autoscale-qwen-model-pvc-pvc-717f9270-0363-4205-b75f-6f118afff3b6/qwen
          type: Directory
      - name: offload-storage
        persistentVolumeClaim:
          claimName: ds-offload-pvc
```

---

## 5. Deployment, start to finish

### 5.1 One-time cluster prep

```bash
# Label new2 so the predictor is pinned there
kubectl label node new2 workload-node=zero-inference

# Confirm GPU is actually schedulable
kubectl describe node new2 | grep nvidia.com/gpu

# Create the local offload directory on new2's root disk
# (no dedicated NVMe on this node — confirmed via df -h)
ssh new2 mkdir -p /mnt/ds-offload

# Tell k3s's containerd this local registry is HTTP, not HTTPS
# (on new2, as root / via sudo)
cat <<EOF | sudo tee /etc/rancher/k3s/registries.yaml
mirrors:
  "10.42.0.1:5000":
    endpoint:
      - "http://10.42.0.1:5000"
configs:
  "10.42.0.1:5000":
    tls:
      insecure_skip_verify: true
EOF
sudo systemctl restart k3s-agent
```

### 5.2 Build the image (on `new2`, to match its exact driver/CUDA)

```bash
cd ~/qwen-deepspeed-runtime
python3 -m py_compile model.py && echo "syntax OK"   # verify before building
sudo docker build -t 10.42.0.1:5000/kserve/qwen-zero-inference:v1 .
sudo docker push 10.42.0.1:5000/kserve/qwen-zero-inference:v1
```

### 5.3 Smoke-test standalone first (before Kubernetes)

```bash
sudo docker run -d \
  --name qwen-zero-test \
  --gpus all \
  -p 8080:8080 \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 \
  -e RANK=0 -e LOCAL_RANK=0 -e WORLD_SIZE=1 \
  -v /mnt/nfs/autoscale-qwen-model-pvc-pvc-717f9270-0363-4205-b75f-6f118afff3b6/qwen:/mnt/models \
  -v /mnt/ds-offload:/local-offload \
  10.42.0.1:5000/kserve/qwen-zero-inference:v1 \
  --http_port=8080 --model_name=qwen-zero-inference

docker logs -f qwen-zero-test
```

Wait for `==== ZERO INFERENCE MODEL READY ====`. This step catches CUDA/DeepSpeed/config problems in a much simpler environment than debugging through Kubernetes pod logs.

### 5.4 Deploy to KServe

```bash
# PV/PVC for the offload path, then the InferenceService itself
kubectl apply -f pv-pvc.yaml
kubectl apply -f isvc.yaml

kubectl get pods -n llm-demo -w
```

If a pod doesn't reach `Running`, check in this order: `kubectl describe pod ...` for scheduling/pull errors first, then `kubectl logs ... --previous` for application crashes. When re-applying `isvc.yaml` after a fix, always check for and delete stale ReplicaSets (`kubectl get rs -n llm-demo`) so you're not reading logs from an old, already-fixed bug.

### 5.5 Confirm it's actually serving and actually swapping

```bash
SERVICE_HOSTNAME=$(kubectl get inferenceservice qwen2-zero-inference -n llm-demo -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -H "Host: ${SERVICE_HOSTNAME}" \
  -H "Content-Type: application/json" \
  http://94.184.177.15:32048/v1/models/qwen2-zero-inference:predict \
  -d '{"instances": [{"prompt": "The capital of France is", "max_new_tokens": 20}]}'
```

While that request runs, confirm real offload activity (not just a successful response):
```bash
ssh new2 iostat -x 1
# or
watch -n 1 'du -sh /mnt/ds-offload'
```

Disk I/O during generation, not just at startup, is the actual confirmation that parameters are streaming in and out of GPU rather than sitting fully resident.

---

## 6. Known constraints and what to revisit before scaling up

- **`new2`'s local disk is limited** (`/dev/vda1`, no dedicated NVMe, modest free space). Fine for 0.5B; a 14B/32B model's offload partitions will need meaningfully more space and ideally faster storage — verify real throughput (`dd`/`fio`) before assuming it'll perform acceptably.
- **`new2` is a shared, memory-constrained cluster node.** The 1Gi/2Gi memory footprint used here works for 0.5B but leaves little margin; a larger model will need either more memory freed up on this node or dedicated headroom reserved for it.
- **The aggressive `stage3_max_live_parameters`/threshold values in `ds_config.json` are tuned to force visible swapping on a model that doesn't need it.** For a model that's genuinely too large for VRAM, these should be relaxed — tight thresholds that make sense for proving the mechanism will hurt throughput on a model that actually needs the capacity.
- **MoE or multimodal architectures are untested** with this setup — ZeRO-3 partitioning assumes fairly standard dense decoder module structure; Mixtral/DeepSeek-MoE-style models have a mixed track record under ZeRO-3 and would need separate validation.
