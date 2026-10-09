import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🍾 Bottle Detection")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image")

    if st.button("Detect Bottles"):
        results = model(image)
        result_image = results[0].plot()

        st.image(result_image, caption="Detection Result", channels="BGR")
        st.write(f"Detected objects: {len(results[0].boxes)}")
