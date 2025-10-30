# Flower Recognition using Deep Learning and Fine-Tuning

A deep learning project for flower classification using CNN and transfer learning with fine-tuning techniques.

## 📋 Project Description

This project implements flower recognition using two approaches:
1. **Custom CNN Model** - Built from scratch for flower classification
2. **Transfer Learning with Fine-Tuning** - Using pre-trained models (VGG16/ResNet/MobileNet) with fine-tuning for improved accuracy

The model can classify 5 types of flowers:
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

## 🚀 Features

- Custom CNN architecture for flower classification
- Transfer learning with fine-tuning
- Interactive GUI application using PyQt6
- Webcam support for real-time prediction
- Training history visualization
- Confusion matrix for model evaluation

## 📁 Project Structure

```
flower-recognition-deep-learning-fine-tuning/
├── CNN_model.py              # Custom CNN model implementation
├── transfer_learning.py      # Transfer learning with fine-tuning
├── app_predict.py           # GUI application for prediction
├── resize.py                # Image preprocessing utilities
├── results/                 # Training results and models
│   ├── results_cnn/        # CNN model results
│   └── results_finetunning/ # Fine-tuned model results
│       ├── flower_finetuned_model.h5
│       ├── training_history_finetune.png
│       └── confusion_matrix_finetune.png
├── .gitignore
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- PyQt6
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Install Dependencies

```bash
pip install tensorflow opencv-python PyQt6 numpy matplotlib seaborn scikit-learn
```

## 📊 Usage

### Training Models

**Train Custom CNN:**
```bash
python CNN_model.py
```

**Train with Transfer Learning and Fine-Tuning:**
```bash
python transfer_learning.py
```

### Run Prediction App

```bash
python app_predict.py
```

The GUI application allows you to:
- Upload images from your computer
- Capture images from webcam
- Get real-time flower classification predictions

### Image Preprocessing

```bash
python resize.py
```

## 🎯 Model Performance

The fine-tuned transfer learning model achieves high accuracy on the flower classification task. Training history and confusion matrix are saved in the `results/results_finetunning/` directory.

## 📝 Dataset

This project uses a flower dataset containing images of 5 flower types. Make sure to organize your dataset in the following structure:

```
data/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
└── test/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

Final Project - Deep Learning Course

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Pre-trained model providers (VGG, ResNet, MobileNet)
- Flower dataset contributors
