# AlphaFold2 with NVIDIA cuEquivariance

This repository provides an implementation of AlphaFold2 that integrates components from [NVIDIA/cuEquivariance](https://github.com/NVIDIA/cuEquivariance) to accelerate equivariant operations commonly used in protein structure prediction. We rely on cuEquivariance-backed kernels across the Evoformer, template embedding, and extra MSA pathways to keep end-to-end inference efficient. Several utility modules are sourced from the [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) project to remain consistent with widely used open-source baselines. The codebase focuses on clarity and modularity while retaining compatibility with PyTorch for experimentation and research.

## Key Features

- **cuEquivariance-accelerated MSA attention** – `MSARowAttentionWithPairBias.forward`, `MSAColumnAttention.forward`, and `MSAColumnGlobalAttention.forward` each dispatch the fused `triangle_attention` kernel for row, column, and global column mixing.
- **Triangle pair attention kernels** – `TriangleAttentionStartingNode.forward` and `TriangleAttentionEndingNode.forward` call the `triangle_attention` operator to keep pair representations synchronized.
- **Fused triangle multiplicative updates** – `TriangleMultiplicationOutgoing.forward` and `TriangleMultiplicationIncoming.forward` leverage `triangle_multiplicative_update` to accelerate outgoing and incoming pair updates.

## Repository Layout

The following functions rely on cuEquivariance kernels to accelerate critical attention and pair-update steps:

- `module/msa.py` – `MSARowAttentionWithPairBias.forward` dispatches the fused `triangle_attention` kernel for row-wise updates, `MSAColumnAttention.forward` leverages the same kernel for column-wise mixing, and `MSAColumnGlobalAttention.forward` shares the kernel for global column mixing.
- `module/triangle_attention.py` – `TriangleAttentionStartingNode.forward` and `TriangleAttentionEndingNode.forward` invoke `triangle_attention` for pairwise attention.
- `module/triangle_multiplication.py` – `TriangleMultiplicationOutgoing.forward` and `TriangleMultiplicationIncoming.forward` call `triangle_multiplicative_update` for fused pair updates.

## Getting Started

1. Install dependencies listed in your preferred environment. A typical setup requires Python 3.10+, PyTorch, `opt_einsum`, and the cuEquivariance package from NVIDIA.
2. Import and instantiate the `Alphafold2` class to prepare the model.
3. Prepare feature dictionaries compatible with AlphaFold-style inputs and invoke the model's `iteration` method for inference.

## License

Refer to the original project licenses for AlphaFold2, NVIDIA cuEquivariance, and the openfold modules incorporated here when distributing derivative work.
