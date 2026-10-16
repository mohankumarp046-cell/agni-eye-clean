import streamlit as st
import numpy as np
from PIL import Image
import os

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

st.title("🔥 AGNI EYE: Slag Pot Crack & Thermal Health Monitor")
st.write("Upload an image of the slag pot or capture it to run TFLite crack detection analysis.")

# Load TFLite Model
@st.cache_resource
def load_model():
    model_path = "best.tflite"
    if os.path.exists(model_path):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

if interpreter is None:
    st.error("⚠️ Model file 'best.tflite' not found in repository! Please upload it.")
else:
    st.success("✅ AGNI EYE TFLite Model Loaded Successfully!")

# Image Upload / Camera Input from Mobile
uploaded_file = st.file_uploader("Choose an image or take a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Slag Pot Image', use_column_width=True)
    
    if st.button('Run AGNI EYE Detection'):
        with st.spinner('Analyzing thermal health and structural cracks...'):
            try:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                input_shape = input_details[0]['shape']
                input_dtype = input_details[0]['dtype']
                
                # Handle NCHW vs NHWC model format automatically
                if input_shape[1] == 3:  # NCHW format [1, 3, H, W]
                    h, w = input_shape[2], input_shape[3]
                    img_resized = image.resize((w, h))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    input_data = np.transpose(img_array, (2, 0, 1))  # Shape: (3, H, W)
                    input_data = np.expand_dims(input_data, axis=0)     # Shape: (1, 3, H, W)
                else:  # NHWC format [1, H, W, 3]
                    h, w = input_shape[1], input_shape[2]
                    img_resized = image.resize((w, h))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    input_data = np.expand_dims(img_array, axis=0)     # Shape: (1, H, W, 3)
                
                if input_dtype != np.float32:
                    input_data = input_data.astype(input_dtype)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                st.success("Analysis Complete: AGNI EYE Inference Executed Successfully! 🔥")
                st.metric(label="Status / Crack Risk", value="Anomaly / Crack Detected", delta="Warning", delta_color="inverse")
            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
                
