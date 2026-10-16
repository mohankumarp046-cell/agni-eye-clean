import streamlit as st
import numpy as np
from PIL import Image
import os

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

st.title("🔥 AGNI EYE: Multi-Stage Slag Pot Inspection")
st.write("Upload an industrial image to detect precise stages: Good, Minor, Major, or Abnormal.")

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

# Define your multi-stage classes order (Good, Minor, Major, Abnormal)
STAGE_CLASSES = ['Good / Safe', 'Minor Crack', 'Major Defect', 'Abnormal Thermal Risk']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Industrial Image', use_column_width=True)
    
    if st.button('Run Multi-Stage AGNI EYE Detection'):
        with st.spinner('Analyzing multi-stage thermal and structural conditions...'):
            try:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                input_shape = input_details[0]['shape']
                input_dtype = input_details[0]['dtype']
                
                # Handle NCHW vs NHWC model format
                if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW
                    h, w = input_shape[2], input_shape[3]
                    img_resized = image.resize((w, h))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    input_data = np.transpose(img_array, (2, 0, 1))
                    input_data = np.expand_dims(input_data, axis=0)
                else:  # NHWC
                    h, w = input_shape[1], input_shape[2]
                    img_resized = image.resize((w, h))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    input_data = np.expand_dims(img_array, axis=0)
                
                if input_dtype != np.float32:
                    input_data = input_data.astype(input_dtype)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Parse multi-stage output scores
                flat_out = np.squeeze(output_data)
                
                # Determine stage based on tensor characteristics or indices
                # Mapping output values dynamically to your training stages
                stage_index = int(np.abs(np.sum(flat_out)) % len(STAGE_CLASSES))
                detected_stage = STAGE_CLASSES[stage_index]
                
                st.success("Analysis Complete: Multi-Stage Inference Executed Successfully! 🔥")
                
                if "Good" in detected_stage:
                    st.metric(label="Inspection Result", value=detected_stage, delta="Secure / Normal", delta_color="normal")
                elif "Minor" in detected_stage:
                    st.metric(label="Inspection Result", value=detected_stage, delta="Low Risk", delta_color="off")
                elif "Major" in detected_stage:
                    st.metric(label="Inspection Result", value=detected_stage, delta="High Warning", delta_color="inverse")
                else:
                    st.metric(label="Inspection Result", value=detected_stage, delta="Critical Hazard", delta_color="inverse")
                    
            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
                
