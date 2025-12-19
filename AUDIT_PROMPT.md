# Audit Request: Independent JAX Codebase Review

## Context
We have recently ported a **PyTorch solution** (Kaggle ASL Fingerspelling 1st Place Solution) to **JAX (Flax NNX)**. The port aims for **1:1 numerical parity** to ensure correctness before further optimizations.
- **Reference PyTorch Model:** `models/mdl_2_pt.py`
- **New JAX Model:** `jax_model/model.py`, `jax_model/layers.py`
- **Verification Script:** `jax_model/verify_parity.py` (Proves ~3.5e-5 logits difference on a sample input).

## Goal
Perform a **comprehensive, independent audit** of the JAX implementation. Do **not** assume the current implementation is correct just because it passes the parity check (e.g., overfitting to one sample, hardcoded fixes, or inefficient patterns).

## Scope & Constraints
1.  **Code Quality & Style:** Ensure code follows Flax NNX best practices and general Python clean code principles.
2.  **No Mocks allowed:** Verify that `jax_model/compat_augmentations.py` and other helpers use **real code** and logic, not mock placeholders.
3.  **Numerical Stability:** Check for potential issues with:
    - Epsilon values in LayerNorm/BatchNorm (PyTorch `1e-5` vs JAX defaults).
    - Precision handling (ensure no implicit float64 slowdowns unless required).
    - Index clamping (e.g., `SinusoidalPositionalEmbedding` indices).
4.  **Efficiency:** Identify patterns that might hinder JIT compilation or XLA optimization (e.g., Python control flow depending on tracers, inefficient tensor slicing/concatenation).
5.  **Hardcoded Values:** Flag any "magic numbers" that should be derived from config (e.g., `33` hardcoded as max length vs `config.max_len`).

## Deliverables
1.  **Audit Report (`audit_report.md`):**
    - **Critical Issues:** Bugs or logic errors that break parity/correctness.
    - **Performance Risks:** XLA-unfriendly patterns.
    - **Code Quality:** Refactoring recommendations.
    - **Coverage Map:** Confirming which parts of `mdl_2_pt.py` are covered by `jax_model/`.
2.  **Refactoring PR (Optional):** If easy fixes are found, apply them, but prioritize **correctness** over style.

## Key Files to Review
- `jax_model/model.py`: Main `Net` class and structure.
- `jax_model/layers.py`: Core layers (`SqueezeformerBlock`, `Decoder`, `LlamaAttention`).
- `jax_model/config.py`: Configuration mapping.
- `jax_model/compat_augmentations.py`: Data augmentation logic (ensure it's legit).

**Instruction for Agent:**
Start by reading `jax_model/model.py` and `jax_model/layers.py` line-by-line. Compare logic against `models/mdl_2_pt.py` where necessary, but focus on **intrinsic correctness** and **JAX/Flax best practices**.
