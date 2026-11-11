# FuseDiff

This is the official implementation of the paper **"Fusion and Diffusion for Timestep-Aware UAV Shadow Removal"**.

## TODO

- [x] Release inference code.
- [x] Release pretrained models.
- [ ] Release training code.

## Overview

Shadow removal in uncrewed aerial vehicle (UAV) imagery faces two key challenges:

- **Lack of explicit coarse-to-fine restoration modeling:** Existing methods often lack explicit control over the progressive transition from global illumination correction to fine-grained texture recovery.

- **Lack of unified stage-wise coordination:** Restoration stages are often processed independently, without a unified mechanism to coordinate illumination correction and texture refinement throughout the restoration process.

FuseDiff consists of three key components:

- **Self-Supervised Inpainting Pretraining:** Learns structural and textural priors from shadow-free images without requiring paired supervision.
- **Timestep-Aware Dynamic Fusion (TADFuse):** Progressively integrates illumination and texture features under timestep guidance.
- **Coarse-to-Fine Diffusion (C2FDiff):** Refines coarse restoration results through conditional diffusion sampling.

![FuseDiff_framework](./FuseDiff_framework.png "FuseDiff_framework")

## Requirements

The implementation is based on Python and PyTorch.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for inference.

## Datasets

The datasets used in our experiments are provided below.

- UAV-SC: [ESC-Net](https://github.com/fzzfRS/ESC_Net)
- SRGTA: [RMMamba](https://github.com/xgd-cj/RMMamba)
- RS-SynShadow: [RS-GSSR](https://fzzfrs.github.io/RS-GSSR/#resources)

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
