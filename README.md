# FuseDiff (UAV Shadow Removal)

This is the official implementation of the paper **"Fusion then Diffusion for Timestep-Aware Semi-Supervised UAV Shadow Removal"**.

## TODO

- [x] Release inference code.
- [x] Release pretrained models.
- [ ] Release training code.

## Overview

Shadow removal in uncrewed aerial vehicle (UAV) imagery remains challenging due to complex illumination variations and fine-grained scene textures. Existing methods often struggle to balance global illumination correction and local texture restoration.

To address these challenges, we propose **FuseDiff**, a timestep-aware framework for UAV shadow removal that combines feature fusion with conditional diffusion refinement.

FuseDiff consists of two key components:

- **Timestep-Aware Dynamic Fusion (TADFuse):** Progressively integrates illumination and texture features under timestep guidance.
- **Coarse-to-Fine Diffusion (C2FDiff):** Refines coarse restoration results through conditional diffusion sampling to recover fine-grained textures and remove residual shadow artifacts.

The framework also incorporates self-supervised inpainting pretraining to reduce its dependence on paired training data.

![FuseDiff_framework](./FuseDiff_framework.pdf "FuseDiff_framework")

## Requirements

The implementation is based on Python and PyTorch.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for inference.

## Datasets

The experiments are conducted on three UAV shadow removal datasets:

- **UAV-SC**
- **SRGTA**
- **RS-SynShadow**

You may also evaluate FuseDiff on your own UAV images. The current framework requires corresponding shadow masks as input.

## Pretrained Models

Please download the pretrained FuseDiff model and place it in the designated checkpoint directory.

- **FuseDiff:** [Download link to be added]

## Inference

To perform shadow removal using the pretrained model, run:

```bash
python <INFERENCE_SCRIPT>.py <REQUIRED_ARGUMENTS>
```

Replace the placeholders with the inference script and arguments provided in the released code.

The default inference configuration employs eight deterministic DDIM sampling steps with $\eta=0$.

The restored shadow-free images will be saved to the specified output directory.

## Results

Qualitative restoration examples and additional experimental results will be provided here.

## Citation

If you find FuseDiff useful in your research, please consider citing our paper.

*BibTeX citation will be added upon publication.*
