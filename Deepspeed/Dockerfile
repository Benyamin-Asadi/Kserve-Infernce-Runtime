# nvcr.io requires auth even for some previously-anonymous public images and
# 403'd during this build — using Docker Hub's mirror instead, no login needed.
FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    libaio-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# libaio-dev is required for DeepSpeed's NVMe async I/O offload engine.
# Skipping this is a silent-failure point: DeepSpeed imports fine and only
# fails once it actually tries to use the nvme offload path at runtime.

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app
COPY requirements.txt .

# Install matching PyTorch build FIRST and explicitly, pinned to the same
# CUDA line as the base image, before deepspeed — deepspeed's build step
# checks torch.version.cuda against nvcc at install time. torch==2.4.1 is
# the actual latest release published on the cu124 wheel index.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Pre-build DeepSpeed's ops at image-build time (ahead-of-time / AOT build)
# instead of JIT-compiling on first request. This moves the CUDA-mismatch
# failure mode to build time, where you can see it and fix it, instead of
# to the first inference request in production.
#
# DS_BUILD_OPS=1 turns ON the AOT build for every op DeepSpeed knows how to
# build on this system. Because DS_BUILD_OPS=1 aborts the ENTIRE install if
# any enabled op fails to compile, every op not needed for single-GPU
# ZeRO-Inference is explicitly disabled below. This list was built against
# real build failures, not assumed:
#   - DS_BUILD_CCL_COMM=0 is required — without it, the build fails on
#     `fatal error: oneapi/ccl.hpp: No such file or directory` (Intel
#     oneAPI CCL headers aren't installed and aren't needed here).
#   - DS_BUILD_SPARSE_ATTN / EVOFORMER_ATTN / TRANSFORMER / etc. are
#     disabled for the same reason: known to abort AOT builds over
#     unrelated, unneeded ops on systems that don't have their specific
#     build dependencies installed.
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

# Verify with DeepSpeed's OWN official verification tool (ds_report), not
# a hand-rolled check. This RUN line must be read by a human at build
# time if it ever changes — a non-error exit does not by itself prove
# async_io built; the grep below checks the report explicitly and fails
# the build if async_io did not build, since that's the one op this
# entire ZeRO-Inference NVMe offload mechanism depends on.
RUN ds_report 2>&1 | tee /tmp/ds_report.log && \
    if grep -qE "async_io.*\[NO\]" /tmp/ds_report.log; then \
        echo "FATAL: async_io op did not build — check libaio-dev is installed and CUDA/torch versions match"; \
        exit 1; \
    fi

COPY model.py ds_config.json ./

ENV MODEL_DIR=/mnt/models
ENV DS_CONFIG_PATH=/app/ds_config.json

ENTRYPOINT ["python", "model.py"]
