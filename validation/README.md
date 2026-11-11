# Validation harness

The scripts behind `docs/VALIDATION.md`. They expect the workspace layout used during the port:

- `weights/bf16/` holds the tier files, as produced by `fetch.py` and `make_modulation.py` from the pinned upstream
  revision.
- `upstream/` holds the upstream configs and examples.
- There are two virtual environments: one with torch 2.15, diffusers ≥ 0.42 (MiniMax-H3-Image support) and
  transformers 5.22 for the reference, and one with MLX.

| file | role |
|---|---|
| `reference.py` | PyTorch reference: diffusers' / transformers' own modules, streamed one block or layer at a time |
| `check_harness.py`, `check_te_harness.py` | prove the streamed reference equals the stock forward passes (max abs diff 0) |
| `pipeline_ref.py`, `edit_ref.py` | end-to-end text-to-image and edit references built from diffusers' helpers |
| `dit_*.py`, `te_*.py`, `vae_*.py`, `vision_parity.py` | stage parity drivers (MLX vs reference) |
| `op_*.py`, `block_floor.py` | bf16 op-level audit and the per-block noise floor across PyTorch-CPU, PyTorch-MPS and MLX |
| `summarize.py` | recompute every published number from the saved dumps |
| `fetch.py`, `make_modulation.py` | selective range download of the needed tensors; the exact AdaLN tables |
