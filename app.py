import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Age Classification using ViT")

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_model()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Age"):
        with st.spinner("Classifying age..."):
            preds = age_classifier(image)
            preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        st.success("Prediction complete")
        st.write(f"**Predicted Age Range:** {preds[0]['label']}")

