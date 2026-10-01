#!/usr/bin/env bash
# Overlay for the validated 20260829 meta-TP release; see the 2026-10-01 report.
# Source before starting a new server. Docker needs explicit -e arguments and
# container recreation; sourcing this does not modify an existing container.
# Keep the model, F16 KV, full context, multimodal and all sampler flags intact.
export GGML_CUDA_ALLREDUCE=hybrid
export GGML_CUDA_ALLREDUCE_HYBRID_MAX_ELEMS=32767
export GGML_CUDA_AR_BF16_THRESHOLD=0
export GGML_CUDA_AR_F16_WIRE=0
export GGML_CUDA_AR_KERNEL_BLOCKS=8
