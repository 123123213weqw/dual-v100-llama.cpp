# Agent build policy

- Do not build CUDA artifacts on a workstation by default.
- The default build host is `WZU_Server`; use `WZU_4080` only for cross-GPU
  comparison rather than V100 acceptance results.
- Sync the repository to `~/codex-build/dual-v100-llama.cpp/repo/` and build
  below `~/codex-build/dual-v100-llama.cpp/`.
- Exclude `.git/`, `.work/`, `build/`, `.env`, model files, and raw prompts from
  synchronization.
- Use `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_CUDA=ON`, and `GGML_NATIVE=OFF` for
  production-comparable V100 builds.
- Never overwrite the running production binary. Build into a versioned
  directory, test on a separate port, then switch the container mount only
  after A/B and correctness checks pass.
- Keep the baseline, safe, and operator artifacts as independent rollback
  targets.
