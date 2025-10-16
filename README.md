Fashion-MNIST Image Classifier (CNN)
Task B: Deep Learning Internship Project

This repository contains the source code, results report, and necessary configuration files for the Deep Learning task involving the classification of the Fashion-MNIST dataset.

The objective was to build a robust Convolutional Neural Network (CNN) to classify 70,000 grayscale images (28x28) into 10 categories of clothing.

Key Results

Final Test Accuracy: 93.50%
Exceeded typical baseline performance for this dataset.

F1-Score (Trouser/Bag): 0.99
Excellent classification of items with distinct shapes.

Weakness: Confusion between Shirt (Class 6) and T-shirt/Top (Class 0).
A common challenge due to low image resolution.

Model Architecture and Strategy

The solution utilizes a deep CNN architecture implemented using TensorFlow/Keras, focusing on stability and generalization.

Core Architecture: Stacked Conv2D layers followed by GlobalAveragePooling and a Dense classification head.

Key Techniques

Batch Normalization (BN): Applied after every convolutional layer to stabilize and accelerate training.

Dropout: Used at 25% in convolutional blocks and 50% in the final dense layer to prevent overfitting.

Learning Rate Scheduler (ReduceLROnPlateau): Dynamically reduced the learning rate when validation loss plateaued, leading to finer convergence and the optimal result at Epoch 46.

Optimization: Adam optimizer with Sparse Categorical Crossentropy loss.

Repository Structure
fashion_mnist_classifier.ipynb / fashion_mnist_classifier.py   →  Complete source code
fashion_mnist_report.md                                         →  Detailed report 
requirements.txt                                                 →  List of all dependencies

How to Run the Project
1. Clone the Repository
git clone [https://github.com/Anjana1-bit/fashion-mnist-classifier.git]
cd fashion-mnist-classifier

2. Install Dependencies

Use a virtual environment and install all required packages:

pip install -r requirements.txt

3. Execute the Code

Open the .ipynb file in a Jupyter environment (Colab/Notebook) and run all cells sequentially

The script will automatically download the dataset, train the CNN, output the final accuracy,
and display the confusion matrix and training history plots.
