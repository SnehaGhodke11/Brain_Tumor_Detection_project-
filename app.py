import os
import gdown
import zipfile
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Google Drive file ID
FILE_ID = "1VsZvdbKFG4muLrXoR29XAYqca7USIVJZ"
ZIP_OUTPUT = "brain_tumor_model.zip"
MODEL_DIR = "model_dir"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")

# Download zip if not exists
if not os.path.exists(ZIP_OUTPUT):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_OUTPUT, quiet=False)

# Extract zip if model not already extracted
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(ZIP_OUTPUT, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Load saved model
model = load_model(MODEL_PATH)

# Class labels (तुझ्या dataset नुसार adjust कर)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor Prediction App")

# Sidebar content
st.sidebar.title("ℹ️ Information")
st.sidebar.subheader("Possible Reasons for Brain Tumor")
st.sidebar.write("""
- Genetic mutations
- Exposure to radiation
- Family history
- Unhealthy lifestyle
- Environmental factors
""")

st.sidebar.subheader("Remedies / Prevention")
st.sidebar.write("""
- Regular medical checkups
- Healthy diet & exercise
- Avoid radiation exposure
- Stress management
- Early diagnosis & treatment
""")

st.sidebar.markdown("---")
st.sidebar.write("👩‍💻 Created by **Sneha Ghodke**")

# Upload or Camera option
option = st.radio("Choose Input Method:", ["Upload File", "Use Camera"])

file_to_use = None
if option == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload MRI Image", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        file_to_use = uploaded_file
elif option == "Use Camera":
    camera_file = st.camera_input("📸 Capture MRI Image")
    if camera_file is not None:
        file_to_use = camera_file

# Predict button
if file_to_use is not None:
    st.image(file_to_use, caption="Selected MRI", use_column_width=True)
    if st.button("🔍 Predict"):
        img = image.load_img(file_to_use, target_size=(128,128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])
        confidence = np.max(preds[0])

        # Threshold logic for "No Tumor"
        if preds[0][2] > 0.6:   # notumor index = 2
            st.success(f"✅ No Tumor Detected (Confidence: {preds[0][2]*100:.2f}%)")
        else:
            st.error(f"⚠️ Tumor Detected: {class_names[pred_class]} (Confidence: {confidence*100:.2f}%)")
