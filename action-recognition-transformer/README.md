# Action Recognition with Transformer

Real-time human action recognition using Vision Transformer (ViT) and YOLOv5 for person detection.

## 📋 Project Description

This project implements real-time action recognition using state-of-the-art deep learning models:
- **Vision Transformer (ViT)** - For action classification
- **YOLOv5** - For person detection and tracking
- **Flask Web App** - For real-time webcam inference

The model can recognize **15 different human actions**:
- Calling
- Clapping
- Cycling
- Dancing
- Drinking
- Eating
- Fighting
- Hugging
- Laughing
- Listening to music
- Running
- Sitting
- Sleeping
- Texting
- Using laptop

## 🚀 Features

- **Vision Transformer (ViT)** architecture for robust action recognition
- **Real-time person detection** using YOLOv5
- **Web-based interface** with Flask for live webcam inference
- **Action smoothing** using temporal filtering
- **High accuracy** (~84% validation accuracy)
- **Data augmentation** for improved generalization
- **Early stopping** and learning rate scheduling

## 📁 Project Structure

```
action-recognition-transformer/
├── notebooks/
│   ├── Final.ipynb           # Complete training pipeline
│   ├── Test.ipynb            # Model testing and evaluation
│   └── Realtime.ipynb        # Real-time inference experiments
├── models/
│   ├── best_vit_model.pth    # Trained ViT model (gitignored)
│   ├── model_ver2.pth        # Alternative model version (gitignored)
│   ├── yolov5s.pt            # YOLOv5 small model (gitignored)
│   └── movenet_lightning.tflite  # MoveNet pose model (gitignored)
├── configs/
│   ├── yolov3-tiny.cfg       # YOLO v3 config
│   ├── yolov3-tiny.weights   # YOLO v3 weights (gitignored)
│   ├── yolov4-tiny.cfg       # YOLO v4 config
│   ├── yolov4-tiny.weights   # YOLO v4 weights (gitignored)
│   └── coco.names            # COCO class names
├── deploy.py                 # Flask web app for deployment
├── webcam.py                 # Webcam inference script
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Webcam (for real-time inference)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Pre-trained Models

Due to file size limitations, model files are not included in the repository. You need to:

1. **Train your own model** using `notebooks/Final.ipynb`
2. **Or download pre-trained models** (if available) and place them in the `models/` directory

## 📊 Usage

### Training

Open and run `notebooks/Final.ipynb` in Jupyter or Google Colab:

```bash
jupyter notebook notebooks/Final.ipynb
```

The notebook includes:
- Dataset preparation and augmentation
- Vision Transformer model configuration
- Training loop with validation
- Model evaluation and visualization
- Confusion matrix and classification report

**Training Configuration:**
- **Model:** ViT-Base (Patch 16, 224x224)
- **Optimizer:** AdamW (lr=3e-5, weight_decay=0.01)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Scheduler:** ReduceLROnPlateau
- **Epochs:** 15 (with early stopping)
- **Batch Size:** 32
- **Dropout:** 0.3

### Real-time Inference with Flask

Run the Flask web application:

```bash
python deploy.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

The web app will:
- Capture video from your webcam
- Detect persons using YOLOv5
- Classify actions using ViT
- Display results in real-time with bounding boxes

### Webcam Inference (Script)

```bash
python webcam.py
```

## 🎯 Model Performance

- **Validation Accuracy:** ~83.87%
- **Test Accuracy:** Available after running evaluation
- **Inference Speed:** ~5 FPS (with person detection)

### Training Results

The model was trained with:
- **Train set:** 12,000 images (800 per class)
- **Validation set:** 3,000 images (200 per class)
- **Test set:** 3,000 images (200 per class)

Training stopped at epoch 6 due to early stopping (best validation accuracy: 83.87%).

## 📝 Dataset Structure

Organize your dataset as follows:

```
data/
├── train_data/
│   ├── calling/
│   ├── clapping/
│   ├── cycling/
│   ├── dancing/
│   ├── drinking/
│   ├── eating/
│   ├── fighting/
│   ├── hugging/
│   ├── laughing/
│   ├── listening_to_music/
│   ├── running/
│   ├── sitting/
│   ├── sleeping/
│   ├── texting/
│   └── using_laptop/
└── test_data/
    └── (same structure as train_data)
```

## 🔧 Configuration

### Model Hyperparameters

Edit in `notebooks/Final.ipynb`:
- `num_classes`: Number of action classes (default: 15)
- `hidden_dropout_prob`: Dropout rate (default: 0.3)
- `learning_rate`: Initial learning rate (default: 3e-5)
- `batch_size`: Training batch size (default: 32)

### Inference Parameters

Edit in `deploy.py`:
- `predict_interval`: Time between predictions (default: 0.2s)
- `confidence_threshold`: Minimum confidence for prediction (default: 0.7)
- `label_history`: Smoothing window size (default: 5)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

Final Project - Deep Learning Course

## 🙏 Acknowledgments

- **Hugging Face Transformers** for Vision Transformer implementation
- **Ultralytics YOLOv5** for person detection
- **timm** (PyTorch Image Models) for model utilities
- **Google ViT** pre-trained models
- Dataset contributors

## 📚 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
