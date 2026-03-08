# Brain Tumor Segmentation using BraTS Dataset

This project explores deep learning approaches for segmenting glioma subregions from multi-modal MRI scans using the BraTS dataset.

The goal is to develop a 3D convolutional neural network capable of identifying tumor regions from volumetric medical images including:

- T1
- T1ce
- T2
- FLAIR

These modalities provide complementary information about tumor structure and are commonly used in clinical neuroimaging.

## Project Objectives

- Understand the structure of multi-modal MRI datasets
- Build preprocessing pipelines for volumetric NIfTI images
- Implement 3D CNN architectures for tumor segmentation
- Address class imbalance using Dice + Cross-Entropy loss
- Evaluate segmentation performance using Dice Score

## Dataset

The dataset used is the **BraTS (Brain Tumor Segmentation) dataset**, which contains multi-modal MRI scans with expert annotations for tumor regions.

Tumor regions include:
- Enhancing tumor
- Tumor core
- Whole tumor

## Current Status

The project is currently under active development.

Work in progress includes:

- MRI data preprocessing
- Loading NIfTI volumes using NiBabel
- Designing the 3D CNN architecture
- Experimenting with segmentation loss functions

## Future Work

- Implement U-Net based 3D architectures
- Experiment with attention mechanisms
- Improve segmentation performance using advanced loss functions
- Evaluate model performance using Dice similarity coefficient

## Tools & Libraries

Python  
PyTorch / TensorFlow  
NiBabel  
NumPy  
OpenCV

---

This project is part of my exploration of **medical image analysis and computational biology**.