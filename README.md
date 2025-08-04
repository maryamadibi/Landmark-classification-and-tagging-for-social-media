# Landmark-classification-and-tagging-for-social-media
This project implements a complete deep learning pipeline to classify landmark images. It covers two main approaches:
Building a Convolutional Neural Network (CNN) from scratch
Applying Transfer Learning using a pretrained model
Additionally, a simple app is created to demonstrate inference on new images.

**Project Overview**
The goal of this project is to design, train, evaluate, and export CNN models for landmark classification. The project is organized into modular code components and Jupyter notebooks to build the pipeline step-by-step:
1. Data loading and preprocessing (src/data.py)
2. Model definition (src/model.py and src/transfer.py)
3. Training and validation (src/train.py, src/optimization.py)
4. Model exporting and inference (src/predictor.py)
5. Interactive application for testing (app.ipynb)

**Repository Contents**
cnn_from_scratch.ipynb:	Notebook for training and evaluating CNN from scratch
transfer_learning.ipynb:	Notebook for transfer learning approach
app.ipynb:	Notebook demonstrating inference with TorchScript model
src/data.py:	Data loading, transformations, and augmentation
src/model.py:	CNN model architecture from scratch
src/transfer.py:	Transfer learning model architecture
src/train.py:	Training and validation loops
src/optimization.py:	Loss function and optimizer definitions
src/predictor.py:	Model inference and TorchScript exporting

**Data Preprocessing**
Data loaders for training, validation, and testing are implemented in src/data.py.
Images are resized to 256 pixels, cropped (random crops for training, center crops for validation/test), converted to tensors, and normalized.
Training set includes data augmentation such as random cropping and potentially other transforms to improve generalization.
Validation and test sets use center cropping for consistency.
Batch sizes, samplers, and worker threads are configurable.
Visualizations of sample batches are provided to verify correct loading and augmentation.

**CNN from Scratch**
Implemented in src/model.py as class MyModel.
Architecture includes convolutional layers, optional dropout (configurable), and fully connected layers.
Output layer size matches the number of landmark classes (num_classes).
The model’s forward method outputs raw logits (softmax is applied later during evaluation).
Reasoning for architectural choices and hyperparameters are discussed in the notebook.

**Transfer Learning**
Implemented in src/transfer.py.
A pretrained backbone model is frozen, and a custom linear classification head is added.
Suitable for leveraging learned features from large-scale datasets.
Training and evaluation follow similar procedures to the CNN from scratch.
Achieves higher accuracy with fewer training epochs.

**Training and Optimization**
Loss function: CrossEntropyLoss (appropriate for multi-class classification).
Optimizers: Both SGD (with momentum) and Adam are supported.
Learning rate schedulers reduce the learning rate when validation loss plateaus.
Training loops include tracking of loss and accuracy on train/validation splits.
Model weights are saved when validation loss improves by more than 1%.

**Model Evaluation**
Validation and test evaluation loops compute loss and accuracy.
Final test accuracy:
CNN from scratch: at least 50% accuracy
Transfer learning model: at least 60% accuracy
Confusion matrices and other metrics are used for detailed analysis.

**Exporting and Deployment**
Models are exported using TorchScript (torch.jit.script) in src/predictor.py.
Exported models are saved as .pt files and reloaded for inference.
An example app (app.ipynb) demonstrates loading a model and running inference on new images outside the training/test sets.

**Suggestions for Improvement and Further Experiments**
Experiment with different augmentation strategies (rotation, color jitter, etc.).
Tune hyperparameters: dropout rate, learning rate, batch size.
Try different architectures to avoid overfitting or improve generalization.
Track experiments and provide tables summarizing losses and accuracy.
Use penultimate layer features to build an image retrieval system based on similarity.
Discuss additional use cases, such as landmark detection in tourism apps or automated image tagging.
