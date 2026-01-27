# 🐱🐶 Cat vs Dog Image Classifier

A deep learning web application that uses Convolutional Neural Networks (CNN) to classify images as either cats or dogs. Built with TensorFlow and deployed as an interactive Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project implements a binary image classification system that distinguishes between cats and dogs with high accuracy. The model is trained on the PetImages dataset and deployed through an intuitive web interface that allows users to upload images and receive instant predictions.

## ✨ Features

- **Real-time Classification**: Upload an image and get instant predictions
- **Confidence Scoring**: See how confident the model is in its prediction
- **User-Friendly Interface**: Clean, modern UI built with Streamlit
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Image Preprocessing**: Automatic image resizing and normalization
- **Model Caching**: Fast predictions with TensorFlow model caching

## 🏗️ Project Structure

```
cat-dog-image-classification/
├── app.py                      # Streamlit web application
├── model_train.ipynb          # Model training notebook
├── model_prediction.ipynb     # Prediction testing notebook
├── cats_vs_dogs_cnn.keras     # Trained CNN model
├── best_sequential_cnn.keras  # Best performing model checkpoint
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                # Git ignore file
└── dataset/
    └── PetImages/            # Training dataset
        ├── Cat/
        └── Dog/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional)

### Installation

1. **Clone the repository** (or download the ZIP):
   ```bash
   git clone https://github.com/yourusername/cat-dog-image-classification.git
   cd cat-dog-image-classification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Model Architecture

The CNN model consists of:

- **Input Layer**: 224x224x3 RGB images
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for spatial dimension reduction
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Single neuron with sigmoid activation for binary classification

### Training Details

- **Dataset**: PetImages (24,997 images)
- **Train/Validation Split**: 80/20
- **Batch Size**: 32
- **Image Size**: 224x224 pixels
- **Normalization**: Pixel values scaled to [0, 1]
- **Framework**: TensorFlow/Keras

## 🎨 Usage

1. **Launch the app** using the command above
2. **Upload an image** using the file uploader (JPG, JPEG, or PNG)
3. **View the prediction** along with confidence score
4. **Download results** (optional)

---
### Dataset

This project uses the Microsoft Cats vs Dogs dataset (PetImages):
- **Total Images**: ~25,000
- **Classes**: Cat (0), Dog (1)
- **Format**: JPG images
- **Source**: Available from Microsoft Research

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **Jupyter**: Interactive development notebooks

## 👤 Author

**Jack**

- LinkedIn: [My LinkedIn Profile](https://linkedin.com/in/johnson-kanyi-2a4209326/)
- GitHub: [rex](https://github.com/jack17kdb)

## 🙏 Acknowledgments

- Microsoft Research for the PetImages dataset
- TensorFlow team for the amazing deep learning framework
- Streamlit team for the intuitive web app framework
- The open-source community for continuous inspiration
