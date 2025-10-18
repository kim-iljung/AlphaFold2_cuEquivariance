# AlphaFold2 with NVIDIA cuEquivariance

This repository provides an implementation of AlphaFold2 that integrates components from [NVIDIA/cuEquivariance](https://github.com/NVIDIA/cuEquivariance) to accelerate equivariant operations commonly used in protein structure prediction. We rely on cuEquivariance-backed kernels across the Evoformer, template embedding, and extra MSA pathways to keep end-to-end inference efficient. Several utility modules are sourced from the [aqlaboratory/openfold](https://github.com/aqlaboratory/openfold) project to remain consistent with widely used open-source baselines. The codebase focuses on clarity and modularity while retaining compatibility with PyTorch for experimentation and research.

## Key Features

- **Evoformer Stack** – A faithful reproduction of the Evoformer architecture with attention, transition, and outer product blocks optimized for cuEquivariance-backed operations.
- **Structure Module** – Predicts backbone and side-chain geometries, leveraging rigid transformations and torsion angle parameterizations.
- **Template and Extra MSA Support** – Embedding pipelines for template features and extra MSA stacks mirror the original AlphaFold2 design while accelerating rigid transformations with cuEquivariance operators.

## Repository Layout

- `module/model.py` – Entry point that orchestrates embedding, Evoformer processing, structure prediction, and auxiliary heads.
- `module/evoformer.py` – Contains the Evoformer stack implementation.
- `module/structure_module.py` – Implements the structure module for geometric reconstruction.
- `module/outer_product_mean.py` – Provides the outer product mean transformation used in pair updates.

## Getting Started

1. Install dependencies listed in your preferred environment. A typical setup requires Python 3.10+, PyTorch, `opt_einsum`, and the cuEquivariance package from NVIDIA.
2. Import and instantiate the `Alphafold2` class to prepare the model.
3. Prepare feature dictionaries compatible with AlphaFold-style inputs and invoke the model's `iteration` method for inference.

## License

Refer to the original project licenses for both AlphaFold2 and NVIDIA cuEquivariance when distributing derivative work.
