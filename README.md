# sMRI-Based Binary Classification using CNN-LSTM

## Project Overview
This project implements a deep learning framework for classifying 3D volumetric sequences, such as medical imaging data, by combining **3D Convolutional Neural Networks (CNNs)** with **Long Short-Term Memory (LSTM)** networks. The architecture incorporates **Squeeze-and-Excitation (SE) blocks** to enhance important channel-wise features, capturing both spatial and temporal dependencies in the data.  

The model is implemented in **PyTorch** and is designed for end-to-end training and evaluation, including batch processing, GPU acceleration, and performance visualization.

---

## Features
- 3D CNNs for spatial feature extraction from volumetric data  
- LSTM layers for temporal sequence modeling  
- Squeeze-and-Excitation (SE) blocks for attention-based feature enhancement  
- Modular design allowing multiple CNN architectures  
- Training and evaluation pipeline with loss and accuracy tracking  
- Visualization of training and validation loss curves  

---

## Preprocessing of Data

The preprocessing pipeline is designed to standardize multi-modal 3D MRI volumes and prepare them for efficient batch loading and training with a 3D CNN–LSTM architecture.

### 1. Input Data Format
- Each patient directory contains multiple **NIfTI (`.nii`) volumes**, corresponding to different MRI modalities.
- The segmentation file (`*seg.nii`) is explicitly ignored during preprocessing.
- Each patient is assigned a binary label:
  - `HGG → 1`
  - `LGG → 0`

---

### 2. Intensity Normalization
All MRI volumes are normalized independently using **min–max normalization**:

x' = (x - min(x)) / (max(x) - min(x) + ε)


- Ensures voxel values lie in the range **[0, 1]**
- Prevents numerical instability using a small epsilon (`1e-8`)
- Reduces inter-scan intensity variations across patients and modalities

---

### 3. Central Slice Extraction
- From the depth dimension (D), only the **central 32 slices** are retained.
- This focuses the model on the most informative region of the brain and reduces computational cost.

```python
start = (D - 32) // 2
end = start + 32
```

---

## CNN Architecture

The spatial feature extraction component of the model is implemented using a **3D Convolutional Neural Network (CNN)**. This CNN processes volumetric MRI data and produces compact feature embeddings that are later modeled temporally using LSTM layers.

Two closely related CNN designs are used:
1. A **baseline 3D CNN**
2. An **Global Pooling CNN**
3. An **SE-enhanced 3D CNN** with channel-wise attention

---

## CNN1. Baseline 3D CNN

### Convolutional Block
Conv3D → BatchNorm → ReLU → MaxPool3D

---

## CNN2: Global Pooling 3D CNN

### Convolutional Block
Conv3D → BatchNorm → ReLU → MaxPool3D

---

## CNN3: CNN2 with Squeeze and Excitation

### Convolution Block
Conv3D → BatchNorm → ReLU → SEBlock → MaxPool3D

## Diagrams

![CNN-1 Architecture](images/diagrams/cnn1.png)

![CNN-2 Architecture](images/diagrams/cnn2_model.png)

![Squeeze-and-Excitation Block](images/diagrams/se_block.png)

![LSTM Architecture](images/diagrams/lstm.png)

---

## Results and Performance Analysis

This section summarizes the training and validation performance of all CNN variants used in the study. The models are compared based on **loss convergence** and **classification accuracy** across epochs.

---

## Individual Model Loss Curves

### CNN-1 (Baseline 3D CNN)
CNN-1 shows strong training convergence with steadily decreasing training loss. However, a noticeable gap between training and validation loss indicates **overfitting**.

![CNN1 Loss Curve](images/results/CNN1_loss_curve.png)

---

### CNN-2 (3D CNN with Global Average Pooling)
CNN-2 demonstrates more stable validation behavior compared to CNN-1. The global average pooling helps regularization, leading to **better generalization**.

![CNN2 Loss Curve](images/results/CNN2_loss_curve.png)

---

### CNN-3 (SE-Enhanced 3D CNN)
CNN-3 exhibits the most balanced training dynamics. The SE blocks improve feature discrimination, resulting in **smoother validation loss** and reduced overfitting.

![CNN3 Loss Curve](images/results/CNN3_loss_curve.png)

---

## Loss Curve Comparison (All Models)

The following plot compares training and validation loss across all CNN variants:

- **CNN-1**: Fast training loss reduction, but poor generalization  
- **CNN-2**: Stable and consistent performance  
- **CNN-3**: Best balance between convergence and validation stability  

![All Models Loss Comparison](images/results/all_models_loss_curve.png)

---

## Accuracy Comparison

Validation accuracy across epochs for all models is shown below:

- CNN-3 achieves the **highest and most consistent accuracy**
- CNN-2 maintains stable accuracy with lower variance
- CNN-1 fluctuates more due to overfitting

![All Models Accuracy](images/results/all_models_accuracy.png)

---

## Summary of Results

| Model | Key Observation |
|-----|----------------|
CNN-1 | Fast convergence, strong overfitting |
CNN-2 | Improved generalization via global pooling |
CNN-3 | Best overall performance with SE attention |

---

## Conclusion:  
The **SE-enhanced CNN (CNN-3)** provides the best trade-off between learning capacity and generalization, making it the preferred spatial feature extractor for the CNN–LSTM pipeline.
