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


# ---- 2. Caption → Story (FLAN‑T5‑LARGE, accurate, imaginative, 50–100 words) ----
@st.cache_resource
def get_story_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

def text2story(caption):
    model = get_story_model()

    prompt = (
        f"Image caption: {caption}\n\n"
        "Write a 50–100 word children's story based on this caption. "
        "The story must stay consistent with the caption, but you may add gentle, imaginative background details "
        "that are not shown in the picture (such as friendly magic, soft sparkles, warm sunshine, or playful sounds). "
        "Do NOT repeat sentences. Do NOT loop phrases. "
        "Make the story cheerful, magical, and easy for young kids.\n\n"
        "Story:"
    )

    output = model(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.92,
        repetition_penalty=3.0
    )[0]["generated_text"]

    story = output.strip()

    # Remove repeated sentences if any
    sentences = story.split(".")
    cleaned = []
    seen = set()

    for s in sentences:
        s = s.strip()
        if len(s) > 0 and s not in seen:
            cleaned.append(s)
            seen.add(s)

    story = ". ".join(cleaned).strip() + "."

    # Enforce 50–100 words
    words = story.split()
    if len(words) < 50:
        story += " Soft sparkles of magic drifted gently through the air, making everything feel warm and full of wonder."
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

