import io
import wave
import random

import librosa
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Cheerful Kids' Image Storytelling App",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Cheerful Kids' Image Storytelling App")
st.write(
    "Upload an image, generate a caption, turn it into a cheerful children's story, "
    "and listen to it as audio."
)

# =========================
# SESSION STATE DEFAULTS
# =========================

defaults = {
    "last_caption": None,
    "last_story": None,
    "last_audio_bytes": None,
    "last_uploaded_name": None,
    "reset_counter": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================
# DEVICE
# =========================

DEVICE = 0 if torch.cuda.is_available() else -1

# =========================
# HUGGING FACE MODELS
# =========================

@st.cache_resource
def get_img2text_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=DEVICE
    )

@st.cache_resource
def get_story_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=DEVICE
    )

@st.cache_resource
def get_tts_model():
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device=DEVICE
    )

# =========================
# HELPERS
# =========================

def img2text(image: Image.Image) -> str:
    model = get_img2text_model()
    output = model(image)
    return output[0].get("generated_text", "").strip()

def build_story_prompt(caption: str, mode: str, voice_style: str, story_seed: int) -> str:
    style_map = {
        "Fairy-tale": "Write it in a soft fairy-tale style.",
        "Adventure": "Write it in a playful adventure style.",
        "Bedtime": "Write it in a calm and comforting bedtime style.",
        "Silly / Funny": "Write it in a silly, funny, laugh-out-loud style.",
        "Superhero": "Write it in a gentle superhero style."
    }

    voice_map = {
        "Friendly narrator": "Use a warm and friendly narrator tone.",
        "Soft bedtime voice": "Use a calm and gentle tone.",
        "Excited storyteller": "Use an energetic and lively tone.",
        "Cartoonish voice": "Use a playful cartoon-like tone."
    }

    prompt = f"""
Image caption: {caption}

Write one brand new children's story of 50 to 100 words based on the image caption above.
The story must clearly follow the caption.
The story must match the selected style.
Make the story cheerful, funny, playful, imaginative, and suitable for young kids.
Use simple English.
Do not repeat sentences.
Do not repeat ideas.
Do not use the same stock magical sentence every time.
Make this version feel fresh and different.
Random story version number: {story_seed}

Selected story mode: {mode}
{style_map.get(mode, "")}
{voice_map.get(voice_style, "")}

Story:
"""
    return prompt.strip()

def clean_story(text: str) -> str:
    raw_sentences = text.replace("\n", " ").split(".")
    cleaned = []
    seen = set()

    for sentence in raw_sentences:
        sentence = " ".join(sentence.strip().split())
        key = sentence.lower()
        if sentence and key not in seen:
            cleaned.append(sentence)
            seen.add(key)

    if not cleaned:
        return ""

    story = ". ".join(cleaned).strip()
    if story and not story.endswith("."):
        story += "."

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."

    return story

def generate_story_once(model, prompt: str) -> str:
    output = model(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3
    )
    return clean_story(output[0].get("generated_text", "").strip())

def text2story(caption: str, mode: str, voice_style: str) -> str:
    model = get_story_model()

    story_seed = random.randint(1000, 999999)
    prompt = build_story_prompt(caption, mode, voice_style, story_seed)
    story = generate_story_once(model, prompt)

    if 50 <= len(story.split()) <= 100:
        return story

    expand_seed = random.randint(1000, 999999)
    expand_prompt = f"""
Image caption: {caption}
Story mode: {mode}

Rewrite the following into one fresh children's story of 50 to 100 words.
Make it cheerful, funny, playful, and suitable for kids.
Keep it consistent with the caption.
Do not repeat any sentence.
Write a new version.
Random rewrite version number: {expand_seed}

Current story:
{story}

Improved story:
"""
    story = generate_story_once(model, expand_prompt)

    if 50 <= len(story.split()) <= 100:
        return story

    final_seed = random.randint(1000, 999999)
    final_prompt = f"""
Caption: {caption}
Mode: {mode}
Version: {final_seed}

Write a completely new children's story in 50 to 100 words.
It must be based on the caption.
It must be cheerful, funny, and imaginative.
It must be different from previous outputs.
No repeated lines.
No filler sentence.
End happily.

Story:
"""
    story = generate_story_once(model, final_prompt)

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."
    return story

def apply_voice_effects(audio: np.ndarray, sr: int, voice_style: str, voice_tone: str) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).squeeze()

    if audio.size == 0:
        raise ValueError("Generated audio is empty.")

    pitch_steps = 0

    if voice_style == "Soft bedtime voice":
        pitch_steps -= 1
    elif voice_style == "Excited storyteller":
        pitch_steps += 1
    elif voice_style == "Cartoonish voice":
        pitch_steps += 2

    if voice_tone == "Lower":
        pitch_steps -= 2
    elif voice_tone == "Higher":
        pitch_steps += 2

    if pitch_steps != 0:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)

    return np.clip(audio, -1.0, 1.0)

def generate_voice_audio(text: str, voice_style: str, voice_tone: str):
    if not text.strip():
        raise ValueError("Story text is empty.")

    tts = get_tts_model()
    output = tts(text)

    if "audio" not in output or "sampling_rate" not in output:
        raise ValueError("TTS model did not return valid audio output.")

    audio = np.asarray(output["audio"], dtype=np.float32).squeeze()
    sr = int(output["sampling_rate"])

    audio = apply_voice_effects(audio, sr, voice_style, voice_tone)
    return audio, sr

def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()

def reset_app():
    st.session_state.last_caption = None
    st.session_state.last_story = None
    st.session_state.last_audio_bytes = None
    st.session_state.last_uploaded_name = None
    st.session_state.reset_counter += 1
    st.rerun()

# =========================
# DYNAMIC WIDGET KEYS
# =========================

suffix = str(st.session_state.reset_counter)

story_mode = st.selectbox(
    "Choose a story mode",
    ["Fairy-tale", "Adventure", "Bedtime", "Silly / Funny", "Superhero"],
    index=0,
    key=f"story_mode_{suffix}"
)

voice_style = st.selectbox(
    "Choose a voice style",
    ["Friendly narrator", "Soft bedtime voice", "Excited storyteller", "Cartoonish voice"],
    index=0,
    key=f"voice_style_{suffix}"
)

voice_tone = st.selectbox(
    "Choose a voice tone",
    ["Neutral", "Lower", "Higher"],
    index=0,
    key=f"voice_tone_{suffix}"
)

st.caption(
    "Note: the voice tone changes the same base Hugging Face TTS voice slightly. "
    "It is not a true male/female speaker switch."
)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{suffix}"
)

# =========================
# IMAGE + GENERATION
# =========================

if uploaded_image is not None:
    if st.session_state.last_uploaded_name != uploaded_image.name:
        st.session_state.last_caption = None
        st.session_state.last_story = None
        st.session_state.last_audio_bytes = None
        st.session_state.last_uploaded_name = uploaded_image.name

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story & Audio", type="primary"):
        try:
            with st.spinner("Generating caption..."):
                caption = img2text(image)

            if not caption:
                st.error("Could not generate a caption from the image.")
            else:
                with st.spinner("Writing cheerful story..."):
                    story = text2story(caption, story_mode, voice_style)

                if not story or len(story.split()) < 30:
                    st.error("Could not generate a good story. Please try again.")
                else:
                    with st.spinner("Creating voice audio..."):
                        voice_audio, sr = generate_voice_audio(story, voice_style, voice_tone)
                        audio_bytes = audio_to_wav_bytes(voice_audio, sr)

                    st.session_state.last_caption = caption
                    st.session_state.last_story = story
                    st.session_state.last_audio_bytes = audio_bytes

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# =========================
# PERSISTENT RESULTS
# =========================

if st.session_state.last_caption:
    st.success("Caption generated!")
    st.write(f"**Caption:** {st.session_state.last_caption}")

if st.session_state.last_story:
    st.success("Story created!")
    st.write("### 📘 Your Story")
    st.write(st.session_state.last_story)

if st.session_state.last_audio_bytes:
    st.success("Audio ready!")
    st.audio(st.session_state.last_audio_bytes, format="audio/wav")
    st.download_button(
        "Download Audio",
        data=st.session_state.last_audio_bytes,
        file_name="kids_story.wav",
        mime="audio/wav"
    )

if st.button("Reset App"):
    reset_app()

if uploaded_image is None:
    st.info("Please upload an image to begin.")
