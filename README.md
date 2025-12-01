# Federated Brain Tumor Detection

A lightweight, privacy-preserving federated deep learning framework for accurate brain tumor detection across distributed medical centers.

---

##  Overview

Brain tumor detection using deep learning often suffers from two major challenges:

* **Insufficient and isolated medical datasets**
* **High computational requirements for training large AI models**

This project provides a **federated, privacy-first, lightweight deep learning solution** that enables multiple medical centers to collaboratively train a shared brain tumor detection model **without sharing raw patient data**.

The global model is further optimized using **pruning** and **quantization** to ensure deployment on regular hospital systems.

---

##  Research Goal

To develop a **lightweight, privacy-preserving federated deep learning framework** that improves brain tumor detection accuracy by training across diverse, distributed MRI datasets.

---

##  Key Features

* **Federated Learning (FL):** Enables collaboration across hospitals without sharing patient data.
* **Custom CNN Model:** Designed specifically for MRI brain tumor classification.
* **Transfer Learning Benchmarks:** ResNet50, VGG16, MobileNet, EfficientNet.
* **Model Compression:** Quantization + pruning to reduce size and speed up inference.
* **Web App Deployment:** A simple interface for uploading MRI scans and receiving predictions.
* **Performance Evaluation:** Accuracy, F1-score, latency, and model-size comparisons.

---

## Motivation

Medical data is sensitive and protected by strict privacy laws. Hospitals rarely share datasets, resulting in small, non‑diverse training data. Federated learning solves this by ensuring that:

* Data **never leaves** the hospital.
* Only **model weights** are shared.
* The model learns from multiple medical centers, improving generalization.

This leads to **more robust, inclusive, and scalable healthcare AI**.

---

## Project Structure

```
📁 federated-brain-tumor-classification-framework
│
├── data/                       # Local dataset partitions (simulated hospitals)
├── federated/                  # Federated learning workflow
│   ├── server.py               # Central server (aggregation)
│   ├── client.py               # Hospital training nodes
│   ├── fedavg.py               # Federated averaging algorithm
│
├── models/
│   ├── custom_cnn.py           # Our custom CNN model
│   ├── mobilenet.py            # Baseline models
│   ├── resnet50.py
│   ├── efficientnet.py
│
├── compression/
│   ├── pruning.py
│   ├── quantization.py
│   ├── evaluate_size.py
│
├── webapp/
│   ├── app.py                  # Flask/Django backend
│   ├── static/
│   ├── templates/
│
├── evaluation/
│   ├── metrics.py              # Accuracy, F1, recall
│   ├── inference_test.py
│
└── README.md
```

---

##  Methodology

### **1. Federated Learning Setup**

* Each dataset (Kaggle MRI datasets) is treated as a separate hospital.
* Each hospital trains locally.
* Only model weights/gradients are shared with the FL server.
* The server aggregates weights using **FedAvg**.

### **2. Custom Model Training**

A lightweight CNN designed for MRI images.

### **3. Transfer Learning Baselines**

Models used for benchmark comparison:

* ResNet50
* VGG16
* MobileNet
* EfficientNet

### **4. Model Compression**

* **Pruning** to remove redundant parameters.
* **Quantization** to reduce memory footprint.

### **5. Web Application Deployment**

User-friendly interface for clinicians to upload MRI scans.

---

## Evaluation Metrics

* Accuracy
* F1-score
* Precision & Recall
* Model size reduction (%)
* Inference time (ms)
* Federated vs centralized performance

---

## Expected Outcomes

* Federated global model that outperforms single-dataset models.
* Reduced model size for deployment on ordinary hospital hardware.
* A privacy-first solution for real-world medical imaging.
* A simple web app for real-time tumor detection.

---

##  Impact

* **Inclusive Healthcare AI**: Hospitals with limited hardware can use AI.
* **Data Privacy**: No raw MRI images leave the hospital.
* **Multi‑Center Collaboration**: Enables AI research without data-sharing risks.
* **Scalable Deployment**: Works across hospitals, universities, and labs.

---

## 👥 Contributors

* *Oyebamiji Mustapha*

---

## License

This project is licensed under the MIT License — free to use and modify.

---

##  Conclusion

This research framework serves as a practical and innovative step toward a secure, collaborative, and lightweight AI solution for brain tumor detection—bridging technical excellence with real-world medical impact.
