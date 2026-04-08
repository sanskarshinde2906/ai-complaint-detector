import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= LOAD =================
model = load_model('model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

MAX_LEN = 88

# ================= FUNCTIONS =================
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='pre')
    
    pred = model.predict(pad)
    class_index = np.argmax(pred)
    
    category = le.inverse_transform([class_index])[0]
    
    return category, pred


def predict_bulk(texts):
    results = []

    for text in texts:
        seq = tokenizer.texts_to_sequences([str(text)])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='pre')
        
        pred = model.predict(pad)
        class_index = np.argmax(pred)
        
        category = le.inverse_transform([class_index])[0]
        results.append(category)

    return results


# ================= UI =================
st.title("🤖 AI Complaint Analyzer")

option = st.radio("Choose Input Type:", ["Single Complaint", "Upload CSV"])

# =========================================================
# 🔹 SINGLE INPUT
# =========================================================
if option == "Single Complaint":
    user_input = st.text_area("Enter your complaint:")

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter a complaint")
        else:
            category, pred = predict_text(user_input)

            st.success(f"📂 Category: {category}")

# =========================================================
# 🔹 CSV INPUT
# =========================================================
elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Data Preview:")
        st.dataframe(df.head())

        # ⚠️ column name change if needed
        text_column = st.selectbox("Select complaint column:", df.columns)

        if st.button("Analyze CSV"):
            predictions = predict_bulk(df[text_column])

            df['Predicted_Category'] = predictions

            st.write("✅ Predictions:")
            st.dataframe(df.head())

            # 🔥 Percentage distribution
            dist = df['Predicted_Category'].value_counts(normalize=True) * 100
            dist = dist.reset_index()
            dist.columns = ['Category', 'Percentage']

            st.write("📊 Category Distribution (%):")
            st.dataframe(dist)

            st.bar_chart(dist.set_index('Category'))