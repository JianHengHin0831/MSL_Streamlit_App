# Malaysian Sign Language Translation Case Study (MySign)

**Course:** WQF7006 - Computer Vision and Image Processing (Semester 1 2025/2026)
**Group:** Occurrence 3 Group 4

### Group Members
* Hin Jian Heng (25069335)
* In Jie Yang (25065486)
* Wang Ruoyu (24082035)
* Lee Wei Quan (25068956)
* Tee Chee Hong (25069322)

---

### Important Resources & Links

**Note:** The complete source code for model training, dataset handling scripts, and the custom MSL dataset are hosted externally. Please refer to the Google Drive link below for these core components.

*   **Model Training Code & Dataset (Google Drive):** https://drive.google.com/drive/folders/1e7yC9Njd8fv92oJLSaZGo17eBwrv2wjY
*   **Deployed Web Application:** https://msl-streamlit-app.pages.dev/
*   **Full Project Report:** Refer to `CV Report.pdf` in this repository.

---

### Overview

This repository contains the deployment files and frontend web application for "MySign", a real-time Malaysian Sign Language (MSL) translation system. The project focuses on classifying 52 discrete MSL glosses using 3D skeletal landmarks. 

Instead of traditional static image classification, this system utilizes Google MediaPipe to extract spatial coordinates (x, y, z) of hands and pose over a sequence of frames. This landmark-based approach reduces computational overhead and preserves user privacy by not transmitting or storing raw video feeds.

### Model Performance

We evaluated four distinct deep learning architectures for processing the sequential landmark data. The models were trained on a custom dataset balanced to 100 samples per class using dynamic transformations and center normalization.

*   **Custom LSTM:** 77.57%
*   **SignLanguageTransformer:** 85.86%
*   **1D-CNN:** 93.00%
*   **SignLanguageTCN (Temporal Convolutional Network):** 94.14% (Best)

The TCN outperformed other architectures by effectively capturing local motion details through dilated convolutions while maintaining a global view of the gesture's entire duration.

### Application Architecture

The web application is built with a serverless architecture to ensure zero network latency during translation.
*   **Inference:** The trained PyTorch TCN model is converted to ONNX format.
*   **Client-Side Execution:** Using `ort.min.js` (ONNX Runtime Web), the inference runs directly on the user's device (CPU/GPU) within the browser.
*   **Features:** Real-time landmark overlay, Text-to-Speech (TTS) integration, adjustable text sizing, and dark/light mode for accessibility.

### Repository Structure

*   `convert_to_onnx.py`: Script used to convert the trained PyTorch (.pth) model into ONNX format for web deployment.
*   `templates/`: Contains the frontend components (HTML, JavaScript, CSS) for the serverless web application.
*   `CV Report.pdf`: Comprehensive project report covering data collection methodology, data engineering, model evaluation, and societal impact analysis.
