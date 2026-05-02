# Program title: Storytelling App

import streamlit as st
from PIL import Image
from transformers import pipeline
import time

# -----------------------------
# Load models once
# -----------------------------
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_tts_model():
    return pipeline("text-to-speech", model="espnet/kan-bayashi-ljspeech_vits")

caption_model = load_caption_model()
tts_model = load_tts_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Image → Caption → Audio Storytelling App")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    with st.spinner("Loading image..."):
        time.sleep(1)
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption & Audio"):
        with st.spinner("Generating caption..."):
            caption = caption_model(image)[0]["generated_text"]

        st.success("Caption generated!")
        st.write(f"**Caption:** {caption}")

        with st.spinner("Generating audio..."):
            audio_output = tts_model(caption)

        audio_array = audio_output["audio"]
        sample_rate = audio_output["sampling_rate"]

        st.audio(audio_array, sample_rate=sample_rate)

