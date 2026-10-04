# Serverless LLM Inference with KServe

This project investigates different approaches for serving large language models in a serverless Kubernetes-based environment using **KServe**. The main focus of this work is the runtime layer: how a model can be executed when its computation or memory requirements exceed the capacity of a single GPU.

Three serving approaches were investigated and implemented:

1. **Tensor Parallelism (TP)** using vLLM
2. **Pipeline Parallelism (PP)** using vLLM
3. **Hierarchical memory offloading** using llama.cpp and DeepSpeed ZeRO-Inference

The experiments were performed using **Qwen2.5-0.5B-Instruct**. Although this model is relatively small, it was used as a controlled test case to evaluate the behavior and implementation of the different serving strategies. The same approaches can be applied to larger models by adjusting their runtime and resource parameters.

---

## 1. Project Overview

The project explores how different inference runtimes can be integrated into a KServe-based serving environment.

The main question is not simply how to run an LLM, but how to distribute its memory and computation when a single GPU is insufficient or when GPU memory needs to be supplemented by other memory tiers.

The project therefore focuses on three different strategies:

### Tensor Parallelism

The model is divided at the tensor level across multiple GPUs. Each GPU holds and processes a portion of the model's tensors, allowing inference to use the combined GPU memory and computational resources.

**Runtime:** vLLM

### Pipeline Parallelism

The model is divided by layers. Different groups of model layers are placed on different GPUs, creating a pipeline through which the input passes.

**Runtime:** vLLM

### Hierarchical Memory Offloading

Instead of keeping the entire model in GPU memory, part of the model is kept in GPU memory while other parts are maintained in CPU memory or NVMe storage and transferred when required.

Two different implementations of this approach were investigated:

* **llama.cpp:** GPU/CPU layer-based model placement with memory mapping.
* **DeepSpeed ZeRO-Inference:** parameter offloading to NVMe using ZeRO Stage 3.

---

# 2. Runtime Implementations

## 2.1 vLLM — Tensor Parallelism

The first vLLM configuration uses **Tensor Parallelism**.

With tensor parallelism, the model computation is distributed across multiple GPUs. Instead of assigning complete layers to individual GPUs, tensors inside the model are partitioned between GPUs.

In this project, two GPUs were used:

```text
                 Model
                   |
          +--------+--------+
          |                 |
       GPU 0             GPU 1
       Tensor            Tensor
       Shards            Shards
          |                 |
          +--------+--------+
                   |
                Output
```

The important vLLM parameter is:

```text
--tensor-parallel-size 2
```

This configures vLLM to use two GPUs for tensor parallel inference.

The experiment also used:

```text
--gpu-memory-utilization 0.75
--max-num-seqs 32
--max-model-len 8192
```

The first parameter controls the fraction of GPU memory that vLLM is allowed to use, while the latter two control the maximum number of concurrent sequences and the maximum model context length.

### Why Tensor Parallelism?

Tensor parallelism is useful when the model cannot efficiently fit within the memory of a single GPU or when multiple GPUs can provide higher inference throughput.

The main trade-off is the communication required between GPUs during inference. Its effectiveness therefore depends not only on the model but also on the GPU architecture and interconnect.

---

## 2.2 vLLM — Pipeline Parallelism

The second vLLM configuration uses **Pipeline Parallelism**.

Instead of splitting tensors within the same layer, pipeline parallelism divides the model into groups of layers and assigns different groups to different GPUs.

For the two-GPU experiment, the conceptual layout is:

```text
Input
  |
  v
+------------------+
|      GPU 0       |
|  Earlier Layers  |
+------------------+
         |
         v
+------------------+
|      GPU 1       |
|   Later Layers   |
+------------------+
         |
         v
      Output
```

The main vLLM parameter is:

```text
--pipeline-parallel-size 2
```

The experiment also used:

```text
--gpu-memory-utilization 0.5
--max-num-seqs 4
--max-model-len 8192
```

Pipeline parallelism provides a different way of distributing the model across GPUs. It can be particularly useful when the model can be divided naturally into layer groups and the complete model cannot fit on a single GPU.

However, communication and pipeline synchronization introduce additional overhead. The resulting performance depends strongly on the model architecture, number of GPUs, workload, and communication characteristics.

---

# 3. llama.cpp — GPU/CPU Layer Offloading

The llama.cpp implementation investigates a different strategy from vLLM.

Instead of distributing the model across multiple GPUs, llama.cpp allows only a selected number of model layers to remain on the GPU while the remaining layers are handled on the CPU side.

For the Qwen2.5-0.5B-Instruct model used in the experiment, the model contains 24 layers. The configuration used:

```text
--n-gpu-layers 10
```

Therefore, the experiment placed approximately 10 layers on the GPU while the remaining layers were kept on the CPU side.

Conceptually:

```text
                Qwen Model
                    |
       +------------+------------+
       |                         |
    GPU Memory              CPU Memory
    10 Layers               14 Layers
       |                         |
       +------------+------------+
                    |
                 Output
```

This approach reduces the amount of GPU memory required by the model.

### GGUF Model Format

The llama.cpp runtime uses a GGUF version of the model:

```text
qwen2.5-0.5b-instruct-q8_0.gguf
```

The model is stored externally and mounted into the runtime through persistent storage.

### Memory Mapping

llama.cpp uses **memory mapping (`mmap`) by default** when loading the model.

Memory mapping allows the model file to be mapped into the process's virtual address space instead of explicitly loading the entire file into RAM at startup. The operating system manages which portions of the mapped file are brought into physical memory as they are accessed.

This is particularly useful for large model files because it avoids requiring the entire model file to be copied into RAM before execution.

`mmap` and GPU layer offloading serve different purposes:

* `mmap` controls how the model file is accessed from storage.
* `--n-gpu-layers` controls how many model layers are placed on the GPU.

The two mechanisms can therefore be used together to create a hierarchical memory configuration.

The experiment kept mmap enabled using llama.cpp's default behavior. The `--no-mmap` option was not used.

---

# 4. DeepSpeed ZeRO-Inference — NVMe Offloading

The third approach investigates more aggressive memory offloading using **DeepSpeed ZeRO-Inference**.

Unlike the llama.cpp configuration, which keeps part of the model on the CPU side, the DeepSpeed runtime was configured to offload model parameters to NVMe storage.

The main configuration uses:

```text
ZeRO Stage 3
```

with parameter offloading:

```text
device: nvme
```

The conceptual memory hierarchy is:

```text
        GPU VRAM
           |
           v
        CPU RAM
           |
           v
        NVMe SSD
```

The goal is to allow inference when the complete model parameters cannot remain in GPU memory.

DeepSpeed manages the movement of parameters between storage and GPU memory as they are required during execution.

The runtime also uses buffers and asynchronous I/O settings to manage data movement between NVMe and memory.

### Custom Runtime

Unlike vLLM and llama.cpp, this implementation required a **custom KServe runtime**.

The custom runtime combines:

* KServe
* Hugging Face Transformers
* PyTorch
* DeepSpeed
* the DeepSpeed configuration
* the model-serving logic

These components are packaged into a dedicated Docker image.

The runtime loads the model, initializes the DeepSpeed engine, receives inference requests through the KServe model interface, performs generation, and returns the generated result.

The DeepSpeed configuration is provided separately through `ds_config.json`, allowing the offloading behavior to be changed without modifying the core serving logic.

---

# 5. Runtime Comparison

The three approaches solve different problems.

| Approach             | Runtime   | Main idea                           | GPUs | Main memory strategy |
| -------------------- | --------- | ----------------------------------- | ---: | -------------------- |
| Tensor Parallelism   | vLLM      | Split tensors across GPUs           |    2 | GPU                  |
| Pipeline Parallelism | vLLM      | Split model layers across GPUs      |    2 | GPU                  |
| Layer Offloading     | llama.cpp | Keep selected layers on GPU         |    1 | GPU + CPU            |
| NVMe Offloading      | DeepSpeed | Offload model parameters to storage |    1 | GPU + CPU + NVMe     |

The approaches should therefore not be considered interchangeable.

**Tensor and pipeline parallelism** primarily distribute computation and model memory across GPUs.

**llama.cpp and DeepSpeed offloading** instead reduce GPU-memory requirements by using lower memory tiers.

The latter approaches sacrifice inference performance in exchange for the ability to execute models under tighter GPU-memory constraints.

---

# 6. Experimental Setup

All experiments used:

```text
Model: Qwen2.5-0.5B-Instruct
```

The runtime configurations were deployed in the KServe environment and tested using HTTP inference requests.

The vLLM experiments used two GPUs, while llama.cpp and DeepSpeed used one GPU each.

The tests were performed under different runtime configurations and workloads, so the results should be interpreted as measurements of the implemented configurations rather than as a perfectly normalized benchmark between the runtimes.

---

# 7. Test Results

## 7.1 vLLM — Pipeline Parallelism

The pipeline-parallel configuration was tested with:

```text
Concurrency: 8
Duration: 60 seconds
Maximum output tokens: 128
Warmup requests: 3
```

Results:

| Metric              |          Result |
| ------------------- | --------------: |
| Total requests      |              88 |
| Successful requests |              88 |
| Failed requests     |               0 |
| Request throughput  |     1.326 req/s |
| Token throughput    | 169.75 tokens/s |
| Mean latency        |       5880.9 ms |
| P50 latency         |       5885.4 ms |
| P95 latency         |       6666.4 ms |
| P99 latency         |       6794.9 ms |

The pipeline-parallel configuration successfully served all requests, demonstrating that the model could be distributed across two GPUs using vLLM's pipeline parallelism.

---

## 7.2 vLLM — Tensor Parallelism

The tensor-parallel configuration used the same general workload:

```text
Concurrency: 8
Duration: 60 seconds
Maximum output tokens: 128
Warmup requests: 3
```

Results:

| Metric              |          Result |
| ------------------- | --------------: |
| Total requests      |             264 |
| Successful requests |             264 |
| Failed requests     |               0 |
| Request throughput  |     4.299 req/s |
| Token throughput    | 548.87 tokens/s |
| Mean latency        |       1847.8 ms |
| P50 latency         |       1878.1 ms |
| P95 latency         |       2119.4 ms |
| P99 latency         |       2290.9 ms |

Under the tested configuration, tensor parallelism achieved substantially higher throughput than pipeline parallelism.

The measured request throughput was approximately **3.24× higher**, while token throughput was approximately **3.23× higher**.

The P50 latency was also substantially lower:

```text
Pipeline Parallelism: 5885.4 ms
Tensor Parallelism:   1878.1 ms
```

These results indicate that tensor parallelism was more effective for this particular workload and hardware configuration.

However, this result should not be interpreted as a general conclusion that tensor parallelism is always superior to pipeline parallelism. Performance depends on factors such as GPU interconnect, model architecture, batch size, sequence length, and parallelization overhead.

---

# 8. llama.cpp Test Results

The llama.cpp configuration used:

```text
Concurrency: 8
Duration: 60 seconds
Maximum output tokens: 128
Warmup requests: 3
GPU layers: 10
```

Results:

| Metric              |         Result |
| ------------------- | -------------: |
| Total requests      |             20 |
| Successful requests |             20 |
| Failed requests     |              0 |
| Request throughput  |    0.218 req/s |
| Token throughput    | 27.92 tokens/s |
| Mean latency        |     32325.0 ms |
| P50 latency         |     35188.3 ms |
| P95 latency         |     38696.4 ms |
| P99 latency         |     38902.8 ms |

The results show significantly lower throughput than the vLLM configurations.

This behavior is expected from the design of the experiment: only part of the model was kept in GPU memory, while the remaining layers were handled on the CPU side. The approach therefore reduces GPU-memory requirements but introduces additional computation and data movement outside the GPU.

The main objective of this configuration was consequently not maximum throughput, but demonstrating that model execution can continue when the complete model is not kept in GPU memory.

---

# 9. DeepSpeed Test Results

The DeepSpeed ZeRO-Inference runtime was tested with:

```text
Concurrency: 1
Duration: 840 seconds
Maximum output tokens: 32
Warmup requests: 1
```

Results:

| Metric              |        Result |
| ------------------- | ------------: |
| Total requests      |             9 |
| Successful requests |             9 |
| Failed requests     |             0 |
| Request throughput  |    0.01 req/s |
| Token throughput    | 0.29 tokens/s |
| Mean latency        |    96079.4 ms |
| P50 latency         |    91717.3 ms |
| P95 latency         |   111392.4 ms |
| P99 latency         |   111392.4 ms |

The DeepSpeed configuration had substantially higher latency and lower throughput than the GPU-resident vLLM configurations.

This is consistent with the purpose of the experiment. The runtime was configured to use NVMe parameter offloading, which introduces storage I/O into the inference path.

The experiment therefore demonstrates the fundamental trade-off of aggressive offloading:

> GPU memory requirements can be reduced by moving model parameters to lower levels of the memory hierarchy, but this comes at a significant inference-performance cost.

The DeepSpeed runtime should therefore be viewed primarily as a **memory-capacity solution**, rather than a high-throughput serving configuration.

---

# 10. Overall Results

The measured results are summarized below.

| Runtime / Method               | GPUs | Requests/s |   Tokens/s |   P50 Latency |
| ------------------------------ | ---: | ---------: | ---------: | ------------: |
| vLLM — Tensor Parallelism      |    2 |  **4.299** | **548.87** | **1878.1 ms** |
| vLLM — Pipeline Parallelism    |    2 |      1.326 |     169.75 |     5885.4 ms |
| llama.cpp — GPU/CPU Offloading |    1 |      0.218 |      27.92 |    35188.3 ms |
| DeepSpeed — NVMe Offloading    |    1 |       0.01 |       0.29 |    91717.3 ms |

The experiments demonstrate four different performance/memory trade-offs.

**Tensor Parallelism** provided the highest throughput in the tested configuration. The model was distributed across two GPUs at the tensor level, allowing both GPUs to participate in inference.

**Pipeline Parallelism** also allowed the model to use multiple GPUs, but its layer-based execution resulted in lower throughput under the tested workload.

**llama.cpp layer offloading** reduced GPU-memory requirements by keeping only a subset of layers on the GPU. This came with a substantial performance cost due to CPU-side execution.

**DeepSpeed NVMe offloading** provided the most aggressive memory offloading strategy. It allowed model parameters to be placed on NVMe storage, but the additional storage I/O resulted in very high latency.

---

# 11. Important Benchmarking Note

These results should not be interpreted as a universal ranking of the runtimes.

The experiments were designed primarily to validate the implementation of different serving strategies. The configurations differed in GPU count, memory strategy, runtime architecture, and workload parameters.

In particular:

* vLLM TP and PP used **two GPUs**.
* llama.cpp used **one GPU** with partial GPU layer placement.
* DeepSpeed used **one GPU** with NVMe parameter offloading.
* DeepSpeed was tested with a lower maximum output length and a single concurrent request.

Therefore, the results demonstrate the behavior and trade-offs of the implemented approaches rather than providing a fully controlled runtime benchmark.

---

# 12. Conclusions

This project demonstrated three different strategies for serving language models under GPU-memory and scalability constraints.

vLLM provided two multi-GPU approaches: Tensor Parallelism and Pipeline Parallelism. Both approaches allow a model to use multiple GPUs, but they distribute the model differently. In the tested environment, Tensor Parallelism achieved significantly better throughput and latency.

llama.cpp demonstrated a different strategy by placing only selected model layers on the GPU and keeping the remaining layers on the CPU side. This reduces GPU-memory requirements and can make model execution possible under constrained GPU resources.

DeepSpeed ZeRO-Inference extended this concept further by using NVMe storage for parameter offloading. This provides access to a larger effective memory capacity but introduces substantial storage-related latency.

Overall, the experiments demonstrate that there is no single serving strategy that is optimal for every deployment scenario. The appropriate approach depends on the available GPU memory, number of GPUs, model size, required throughput, and acceptable latency.

For **high-throughput multi-GPU inference**, tensor parallelism was the most effective approach among the tested configurations. For **GPU-memory-constrained inference**, layer-based or NVMe-based offloading provides an alternative when keeping the entire model in GPU memory is not practical.

The project also demonstrated that these different inference engines can be integrated into a common KServe-based serving environment, including the creation of a custom runtime when an existing serving runtime does not provide the required functionality.
