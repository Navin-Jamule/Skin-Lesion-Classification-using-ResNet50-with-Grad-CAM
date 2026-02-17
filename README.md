# Skin-Lesion-Classification-using-ResNet50-with-Grad-CAM
Web-based decision support system for classifying dermoscopic skin images as benign or malignant using a fine-tuned ResNet50 deep learning model. The application provides prediction confidence and Grad-CAM visual explanations to highlight lesion-relevant regions. Developed with PyTorch and deployed using Streamlit for real-time image analysis.

---

##  Features

-  Upload dermoscopic skin image
-  Binary classification (Benign / Malignant)
-  Prediction confidence score
-  Grad-CAM heatmap visualization
-  Interactive Streamlit web interface

---

##  Model Details

- Architecture: ResNet50
- Transfer Learning with fine-tuning (Layer3 & Layer4)
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate Scheduler: StepLR
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - ROC-AUC

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- NumPy
- Matplotlib

---

## Output

<img width="1920" height="3077" alt="Image" src="https://github.com/user-attachments/assets/05bca4cc-b5fd-4f3f-9d8a-00e37230d92a" />


