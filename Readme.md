# 🤖 AI Complaint Detector

An NLP-based web application that classifies customer complaints into predefined categories using a Deep Learning model.

---

## 🚀 Overview

This project uses Natural Language Processing (NLP) techniques along with a trained TensorFlow/Keras model to analyze and categorize user-input complaints. The application is built with Streamlit to provide a simple and interactive interface.

## 🧠 Model Capabilities

This model classifies customer complaints into the following 5 categories:

- Credit Reporting
- Debt Collection
- Mortgages and Loans
- Credit Card Issues
- Retail Banking

The model is trained using NLP techniques including tokenization and sequence padding.

---

## 🧠 Key Features

- Classifies complaint text into categories
- Uses trained deep learning model
- Real-time prediction
- Clean and minimal user interface

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas

---

## ⚙️ Model Details

- Text preprocessing using Tokenizer
- Sequence padding for fixed input length
- Trained classification model (`.h5`)
- Label encoding for category output

---

## 📂 Project Structure

AI COMPLAIN DETECTOR/
│── app.py  
│── model.h5 ❌  
│── tokenizer.pkl ❌  
│── label_encoder.pkl ❌  
│── requirements.txt  
│── README.md  

---

## ⚠️ Note

Model and tokenizer files are not included in this repository due to size constraints.

---

## 💡 Future Improvements

- Improve model accuracy
- Add more categories
- Enhance UI/UX
- Deploy as a live web app