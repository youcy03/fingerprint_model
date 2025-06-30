# 🧠 Fingerprint Recognition using Deep Learning and Flask

This is my final year project for fingerprint recognition using Convolutional Neural Networks (CNN) and Grad-CAM visualization. The app is built using Python and deployed with Flask.

## 📁 Project Structure

fingerprint_model/
├── app/
│ ├── app.py
│ ├── utils.py
│ ├── static/ (icons, CSS, uploads...)
│ ├── templates/ (HTML frontend)
│ └── model/ (trained model and encoder)
├── training/
│ └── notebook.ipynb (model training notebook)
├── requirements.txt
├── README.md


## 🔧 Features

- CNN model trained on SOCOFing dataset
- Flask web interface to upload fingerprint images
- Prediction of user ID from fingerprint
- Grad-CAM visualization for model explainability
- Custom UI with sounds and feedback icons

## 🚀 Installation

```bash
pip install -r requirements.txt
cd app
python app.py
```
## 📸 Preview

### Homepage
![App Screenshot](app.png)

### Prediction Result
![Prediction Screenshot](app_correct.png)

🧠 Tech Stack
Python

TensorFlow / Keras

Flask

OpenCV

Grad-CAM

HTML/CSS

## 👩‍💻 Author

**Bachri Yousra**  
[GitHub](https://github.com/youcy03) – [Email](mailto:yousra.bachri03@gmail.com)
