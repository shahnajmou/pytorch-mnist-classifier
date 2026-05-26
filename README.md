# Handwritten Digit Classification using PyTorch

This is a beginner-friendly deep learning project that classifies handwritten digits using the MNIST dataset and PyTorch.

## Project Overview

The goal of this project is to train a neural network to recognize handwritten digits from 0 to 9. The model is trained on the MNIST dataset, which contains grayscale images of handwritten digits.

## Technologies Used

- Python
- PyTorch
- TorchVision
- Matplotlib

## What I Learned

- How to load image datasets using TorchVision
- How to convert images into tensors
- How to build a neural network using PyTorch
- How to train and evaluate a deep learning model
- How to calculate model accuracy
- How to save a trained model

## Model Architecture

The model is a simple fully connected neural network:

```text
Input Image 28x28
→ Flatten Layer
→ Linear Layer
→ ReLU
→ Linear Layer
→ ReLU
→ Output Layer with 10 classes
Results

The model achieved high accuracy on the MNIST test dataset.

Example output:

Test Accuracy: 97.36%

How to Run

Clone the repository:

git clone your-repository-link
cd pytorch-mnist-classifier

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the project:

python mnist_classifier.py


## CNN Upgrade

I also upgraded the original fully connected neural network into a Convolutional Neural Network (CNN).

The CNN architecture includes:

- Convolutional layers for feature extraction
- ReLU activation functions
- Max pooling layers for dimensionality reduction
- Fully connected layers for final digit classification

CNN models are more suitable for image classification because they can learn spatial features such as edges, curves, and shapes.

## CNN Results

CNN Test Accuracy: 98.99%

pip install -r requirements.txt

Run the project:

python mnist_classifier.py
