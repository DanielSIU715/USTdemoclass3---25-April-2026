import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import tempfile
import wave
import io

# =========================
# MAIN UI
# =========================

st.title("📖 Cheerful Kids' Image Storytelling App")
st.write("Upload an image → get a caption → generate a magical kids story → listen to a friendly voice.")

# =========================
# FUNCTION PART
# =========================

# ---- 1. Image → Caption (BLIP) ----
@st.cache_resource
def get_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def img2text(image):
    model = get_img2text_model()
    return model(image)[0]["generated_text"]


# ---- 2. Caption → Story (FLAN‑T5, accurate, kid‑friendly, 50–100 words) ----
@st.cache_resource
def get_story_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def text2story(caption):
    model = get_story_model()

    prompt = (
        f"Image caption: {caption}\n\n"
        "Write a 50–100 word children's story based strictly on this caption. "
        "Do not add unrelated objects or events. "
        "Make the story cheerful, magical, and easy for young kids. "
        "Add gentle fantasy elements only if they fit naturally.\n\n"
        "Story:"
    )

    output = model(
        prompt,
        max_new_tokens=180,
        temperature=0.5,
        top_p=0.9
    )[0]["generated_text"]

    # Clean up
    story = output.strip()
    words = story.split()

    # Enforce 50–100 words
    if len(words) < 50:
        story += " Soft sparkles of magic floated gently around, making the moment feel warm and full of wonder."
    elif len(words) > 100:
        story = " ".join(words[:100]) + "."

    return story


# ---- 3. Story → Voice (Hugging Face TTS) ----
@st.cache_resource
def get_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

def generate_voice_audio(text):
    tts = get_tts_model()
    out = tts(text)
    return out["audio"], out["sampling_rate"]


# ---- 4. Save audio to WAV ----
def save_audio(audio, sr):
    audio_int16 = (audio * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        with wave.open(fp.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio_int16.tobytes())
        return fp.name


# =========================
# MAIN APP LOGIC
# =========================

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Story"):
        with st.spinner("Generating caption..."):
            caption = img2text(image)
        st.success("Caption generated!")
        st.write(f"**Caption:** {caption}")

        with st.spinner("Writing cheerful story..."):
            story = text2story(caption)
        st.success("Story created!")
        st.write("### 📘 Your Story")
        st.write(story)

        with st.spinner("Creating friendly voice..."):
            voice_audio, sr = generate_voice_audio(story)

        audio_path = save_audio(voice_audio, sr)

        st.success("Audio ready!")
        st.audio(audio_path)

    if st.button("Clear"):
        st.experimental_rerun()
