## Semantic Segmentation using Unet from scratch and pre-trained models

In this project, I performed semantic segmentation on the Cityscapes dataset, which contains high-quality annotated images for urban scene understanding. Semantic segmentation involves generating classification labels for each pixel in an image to understand its content. I used two approaches:

1. **Own Model**: Designed a U-Net architecture and trained it from scratch using the given data.
2. **Pre-trained Model**: Used EfficientNet-B3 as the backbone architecture within a DANet (Dual Attention Network) framework, leveraging the TensorFlow Advanced Segmentation Models (TASM) library.

This report outlines the input and output formats, data analysis, algorithm effectiveness, and visualization results.

## Task Input and Output

### Input
- **Data Format**: Images from the Cityscapes dataset, resized to 256×256 dimensions for uniformity.
- **Channels**: RGB images with three color channels.
- **Labels**: Pixel-wise annotations indicating class categories such as roads, buildings, pedestrians, and vehicles.

### Output
- **Segmentation Masks**: Pixel-wise predictions with the same resolution as the input images.
- **Visualization**: Training and validation loss for both the U-Net and pre-trained model, along with actual images, ground truth labels, and predicted labels.

## Data Analysis

The Cityscapes dataset includes:
- **Training Set**: 2,975 images
- **Validation Set**: 500 images
- **Test Set**: 1,525 images
- **Class Distribution**: 34 classes such as "road," "building," "vegetation," "car," and "pedestrian." The dataset is imbalanced, with dominant classes like "road" and "building."

[TensorFlow Advanced Segmentation Models Repository](https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models?tab=readme-ov-file)

## Preprocessing

- Resized images to 256×256.
- Normalized pixel values to range [0,1].
- Augmented data using flipping, rotation, and brightness adjustment for model robustness.

## Algorithm and Visualization Analysis

### Experiment 1: U-Net Trained from Scratch

**Model Architecture**:
- **Encoder-Decoder Structure**: Skip connections for better feature propagation.
- **Encoder**: Extracts spatial features using convolutional and pooling layers.
- **Decoder**: Reconstructs segmentation maps using upsampling and convolutional layers.

**Training Details**:
- **Optimizer**: Adam (default learning rate)
- **Loss Function**: Categorical Cross-Entropy
- **Epochs**: 50
- **Batch Size**: 16

**Results**:
- **Training Accuracy**: 96.01%
- **Validation Accuracy**: 81.14%

**Visualization**:
- **Loss Visualization**: The training loss decreases consistently, but validation loss fluctuates, indicating potential overfitting.
- **Actual Segmentation Visualization**: Displays the original image, input mask (ground truth), and predicted mask side by side.

### Experiment 2: Pre-trained Model

**Model Details**:
- **Framework**: Dual Attention Network (DANet)
- **Backbone**: EfficientNet-B3
- **Weights**: Pre-trained on ImageNet, fine-tuned on Cityscapes
- **Implementation**: TensorFlow Advanced Segmentation Models (TASM) library

**Training Details**:
- **Optimizer**: SGD (learning rate = 0.2, momentum = 0.9)
- **Loss Function**: Categorical Focal Loss
- **Epochs**: 10
- **Metrics**: IoU score with threshold 0.5
- **Batch Size**: 16

**Results**:
- **Training Loss**: 0.0063
- **Validation Loss**: 0.0086
- **Training IoU Score**: 0.5458
- **Validation IoU Score**: 0.5047

**Visualization**:
- **IoU Score and Loss Visualization**: Training IoU improves steadily, but validation IoU fluctuates, suggesting sensitivity to certain data characteristics.

## Conclusion

The **Dual Attention Network (DANet) with EfficientNet-B3 backbone**, fine-tuned on the Cityscapes dataset, was trained for **10 epochs** due to its resource-intensive nature. The training was conducted on **Kaggle's P100 GPU**.

- **DANet achieved a training IoU score of 0.5458**, with low training and validation losses.
- **U-Net, trained from scratch for 50 epochs**, reached **96.01% training accuracy**, but the **81.14% validation accuracy** suggests overfitting.
- Given resource constraints, both models showed effective segmentation performance, with DANet providing a good balance of accuracy and efficiency within available resources.

