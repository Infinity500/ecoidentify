import streamlit as st
from roboflow import Roboflow
import tempfile

rf = Roboflow(api_key="Os2ZluVVWRriNOylgb1Z")
project = rf.workspace().project("ecoidentify-pubxp")
model = project.version("1").model
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

st.title("EcoIdentify-Roboflow API")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    result = model.predict(temp_path).json()
    st.write("Result:", result["predictions"][0]["predictions"][0]["class"])

    for prediction in result['predictions'][0]['predictions']:
        st.write(f"**Confidence: {prediction['confidence']*100}%**")
        st.progress(int(prediction['confidence'] * 100))
