# Button Cell Classifier

A deep learning-based image classification system that identifies button cell batteries from different views (top, side, and bottom) using a fine-tuned ResNet50 model.

## Project Overview

This project implements a computer vision solution to classify button cell batteries based on their orientation. It uses transfer learning with a pre-trained ResNet50 model that has been fine-tuned on a custom dataset of battery images from three different viewpoints. The final model can be used to classify images or for real-time classification using a webcam.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Real-time Classification](#real-time-classification)
- [Usage](#usage)

## Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements include:
- torch>=1.10.0
- torchvision>=0.11.0
- numpy>=1.19.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- pillow>=8.0.0
- opencv-python>=4.5.0

## Project Structure

- `Battery.ipynb`: Main notebook containing the complete pipeline
- `button_cell_classifier.pth`: Trained model weights
- `requirements.txt`: Required packages for the project
- `button_cells` : Dataset

## Dataset

The dataset consists of button cell battery images from three different views:
- Top view
- Side view
- Bottom view

The dataset is preprocessed to:
1. Remove corrupt images
2. Apply data augmentation (random crops, flips, rotations, and color jitter)
3. Normalize using ImageNet statistics
4. Split into training (80%) and validation (20%) sets

## Model Architecture

The project uses the **ResNet50** architecture, which is a 50-layer deep Convolutional Neural Network (CNN) with residual connections. ResNet50 was chosen for its:

- Strong performance on image classification tasks
- Ability to learn deep representations without gradient vanishing problems due to residual connections
- Pre-trained weights on ImageNet that provide a good starting point for transfer learning

The model architecture has been modified by:
1. Keeping all pre-trained layers
2. Replacing the final fully connected layer to output 3 classes (top, side, bottom views)
3. Fine-tuning all layers during training

## Training Pipeline

The training pipeline consists of the following components:

### 1. Data Preparation
- Data cleaning to remove corrupt images
- Data loading using PyTorch's `ImageFolder`
- Class balancing using `WeightedRandomSampler` to handle class imbalance

### 2. Data Augmentation
```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

### 3. Model Training
- Fine-tuning a pre-trained ResNet50 model
- Using Adam optimizer with a learning rate of 1e-5
- Training for 20 epochs with early stopping based on validation accuracy
- Saving the best model based on validation accuracy

### 4. Training Process
Each training epoch includes:
- Forward pass through the model
- Loss calculation using Cross-Entropy Loss
- Backward pass and parameter updates
- Training and validation accuracy/loss recording
- Model saving when improved validation accuracy is achieved

## Evaluation

The model is evaluated using multiple metrics:

### 1. Performance Metrics
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

### 2. Visualization
- Training and validation accuracy curves
- Training and validation loss curves
- Visualization of misclassified images for error analysis

### 3. Confusion Matrix
The confusion matrix shows:
- True positives along the diagonal
- False positives and false negatives off the diagonal
- The model's performance across all classes

## Real-time Classification

The project includes a real-time classification component using a webcam:

1. Opens the default camera
2. Processes each frame:
   - Center crops the image
   - Applies the same transformations used during training
   - Feeds the processed image to the model
3. Displays the prediction on the video feed
4. Closes when 'q' is pressed

```python
# Example of the real-time classification code
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Process frame and make prediction
    # Display result
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Usage

### For single image prediction:

```python
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("button_cell_classifier.pth"))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Predict function
def predict_image(image_path, model, transform):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    
    return class_names[pred.item()]

# Example usage
class_names = ['bottom_view', 'side_view', 'top_view']
prediction = predict_image("path/to/image.jpg", model, transform)
print(f"Prediction: {prediction}")
```

### For real-time classification:

Run the webcam code section in the Jupyter notebook to start real-time classification.
