import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import tempfile
import wave
import requests
import io

# =========================
# FUNCTION PART
# =========================

# ---- 1. Image → Caption (BLIP) ----
@st.cache_resource
def get_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def img2text(image):
    model = get_img2text_model()
    caption = model(image)[0]["generated_text"]
    return caption


# ---- 2. Caption → Story (GPT‑2, cheerful kids version, 50–100 words) ----
@st.cache_resource
def get_story_model():
    return pipeline("text-generation", model="gpt2")

def text2story(caption):
    model = get_story_model()

    prompt = (
        "You are a joyful children's storyteller. Write a cheerful, magical, and fun story "
        "between 50 and 100 words based on the following image description: "
        f"'{caption}'. "
        "The story must be lively, imaginative, and suitable for young kids. "
        "Use simple, happy language and add a sense of wonder. "
        "Avoid repeating sentences or phrases. End the story with a warm, positive feeling.\n\nStory:"
    )

    output = model(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.75,
        top_p=0.9,
        repetition_penalty=2.0
    )[0]["generated_text"]

    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    sentences = output.split(".")
    cleaned = ". ".join(s.strip() for s in sentences if len(s.strip()) > 0)
    cleaned = cleaned.strip() + "."

    return cleaned


# ---- 3A. Story → Voice (Hugging Face TTS) ----
@st.cache_resource
def get_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

def generate_voice_audio(text):
    tts = get_tts_model()
    out = tts(text)
    return out["audio"], out["sampling_rate"]


# ---- 3B. Load cheerful background music (MP3 from GitHub RAW URL) ----
@st.cache_resource
def get_mp3_decoder():
    return pipeline("audio-classification", model="hf-audio/encodec_32khz")

def load_music_from_github():
    url = "https://raw.githubusercontent.com/DanielSIU715/USTdemoclass3---25-April-2026/main/assets/cheerful_music.mp3"

    response = requests.get(url)
    audio_bytes = io.BytesIO(response.content)

    decoder = get_mp3_decoder()
    decoded = decoder(audio_bytes)

    audio = decoded["audio"]
    sr = decoded["sampling_rate"]

    return audio, sr


# ---- 3C. Mix voice + music ----
def mix_audio(voice, voice_sr, music, music_sr, music_volume=0.25):
    if music_sr != voice_sr:
        factor = music_sr / voice_sr
        step = max(int(round(factor)), 1)
        music = music[::step]

    if len(music) < len(voice):
        repeats = int(np.ceil(len(voice) / len(music)))
        music = np.tile(music, repeats)
    music = music[:len(voice)]

    mixed = voice + music_volume * music

    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val

    return mixed, voice_sr


# ---- 3D. Save audio to WAV ----
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
# MAIN PART (Streamlit UI)
# =========================

st.title("📖 Cheerful Kids' Image Storytelling App")
st.write("Upload an image → get a caption → generate a magical kids story → listen to a friendly voice with cheerful background music.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
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

        with st.spinner("Adding cheerful background music..."):
            music_audio, music_sr = load_music_from_github()
            mixed_audio, mixed_sr = mix_audio(voice_audio, sr, music_audio, music_sr)

        audio_path = save_audio(mixed_audio, mixed_sr)

        st.success("Audio ready!")
        st.audio(audio_path)

    if st.button("Clear"):
        st.experimental_rerun()




