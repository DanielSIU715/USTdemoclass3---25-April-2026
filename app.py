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
    "Upload an image, generate a caption, create a cheerful children's story, "
    "and listen to it as audio."
)

# =========================
# SESSION STATE DEFAULTS
# =========================

defaults = {
    "uploaded_image_bytes": None,
    "uploaded_image_name": None,
    "last_caption": None,
    "last_story": None,
    "last_audio_bytes": None,
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
# IMAGE / STORY HELPERS
# =========================

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def img2text(image: Image.Image) -> str:
    model = get_img2text_model()
    output = model(image)
    return output[0].get("generated_text", "").strip()

def get_style_instruction(mode: str) -> str:
    style_map = {
        "Fairy-tale": "Write it as a magical fairy tale with wonder, charm, and a happy ending.",
        "Adventure": "Write it as a playful adventure with discovery and excitement.",
        "Bedtime": "Write it as a calm bedtime story with a peaceful and cozy feeling.",
        "Silly / Funny": "Write it as a silly and funny story that may make kids laugh.",
        "Superhero": "Write it as a gentle superhero story with courage, kindness, and fun."
    }
    return style_map.get(mode, "Write it as a cheerful children's story.")

def get_voice_instruction(voice_style: str) -> str:
    voice_map = {
        "Friendly narrator": "Use a warm and friendly storytelling tone.",
        "Soft bedtime voice": "Use a soft and gentle storytelling tone.",
        "Excited storyteller": "Use an energetic and lively storytelling tone.",
        "Cartoonish voice": "Use a playful and cartoon-like storytelling tone."
    }
    return voice_map.get(voice_style, "Use a cheerful storytelling tone.")

def remove_prompt_echo_sentences(sentences):
    banned_phrases = [
        "story mode",
        "selected mode",
        "picture description",
        "image caption",
        "caption:",
        "prompt",
        "instruction",
        "story version",
        "rewrite version",
        "the story mode is",
        "write one brand new",
        "write a fresh",
        "rules:"
    ]

    filtered = []
    for sentence in sentences:
        s = sentence.strip()
        low = s.lower()
        if s and not any(phrase in low for phrase in banned_phrases):
            filtered.append(s)
    return filtered

def clean_story(text: str) -> str:
    text = text.replace("\n", " ").strip()
    raw_sentences = text.split(".")
    cleaned = []
    seen = set()

    for sentence in raw_sentences:
        sentence = " ".join(sentence.strip().split())
        normalized = sentence.lower()
        if sentence and normalized not in seen:
            cleaned.append(sentence)
            seen.add(normalized)

    cleaned = remove_prompt_echo_sentences(cleaned)

    if not cleaned:
        return ""

    story = ". ".join(cleaned).strip()
    if not story.endswith("."):
        story += "."

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."

    return story

def generate_story_once(prompt: str, max_new_tokens: int = 180) -> str:
    model = get_story_model()
    output = model(
        prompt,
        max_new_tokens=max_new_tokens,
        min_length=50,
        do_sample=True,
        temperature=0.95,
        top_p=0.95,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3
    )
    return clean_story(output[0].get("generated_text", "").strip())

def build_main_story_prompt(caption: str, mode: str, voice_style: str) -> str:
    style_instruction = get_style_instruction(mode)
    voice_instruction = get_voice_instruction(voice_style)
    story_seed = random.randint(1000, 999999)

    prompt = f"""
Picture description: {caption}

Write one brand new children's story in 50 to 100 words.

Requirements:
- Use the main nouns, objects, animals, and actions from the picture description.
- Keep the story clearly related to the picture description.
- {style_instruction}
- {voice_instruction}
- Make it cheerful, imaginative, playful, and suitable for young kids.
- Use simple English.
- Do not repeat sentences.
- Do not mention labels like picture description, caption, prompt, instruction, or story mode.
- End with a happy or warm feeling.

Story version: {story_seed}

Story:
"""
    return prompt.strip()

def build_expand_prompt(caption: str, mode: str, voice_style: str, short_story: str) -> str:
    rewrite_seed = random.randint(1000, 999999)
    style_instruction = get_style_instruction(mode)
    voice_instruction = get_voice_instruction(voice_style)

    prompt = f"""
Picture description: {caption}

Here is a short children's story:
{short_story}

Rewrite and expand it into one complete children's story in 50 to 100 words.

Requirements:
- Keep it clearly related to the picture description.
- Keep the same core meaning, but add more cheerful and imaginative details.
- {style_instruction}
- {voice_instruction}
- Use simple English for kids.
- Do not repeat sentences.
- Do not mention prompts, labels, caption, or story mode.
- End with a happy or warm feeling.

Rewrite version: {rewrite_seed}

Improved story:
"""
    return prompt.strip()

def build_retry_prompt(caption: str, mode: str, voice_style: str) -> str:
    retry_seed = random.randint(1000, 999999)
    style_instruction = get_style_instruction(mode)
    voice_instruction = get_voice_instruction(voice_style)

    prompt = f"""
Picture description: {caption}

Write a completely new children's story in 50 to 100 words.

Requirements:
- The story must be clearly based on the picture description.
- Use the important nouns and actions from the picture description.
- {style_instruction}
- {voice_instruction}
- Make it cheerful, imaginative, playful, and suitable for kids.
- Use simple English.
- Do not copy the picture description as a single sentence.
- Do not repeat sentences.
- Do not mention prompts, labels, caption, or story mode.
- End happily.

New story version: {retry_seed}

Story:
"""
    return prompt.strip()

def text2story(caption: str, mode: str, voice_style: str) -> str:
    story = generate_story_once(
        build_main_story_prompt(caption, mode, voice_style),
        max_new_tokens=180
    )

    if 50 <= len(story.split()) <= 100:
        return story

    story = generate_story_once(
        build_expand_prompt(caption, mode, voice_style, story),
        max_new_tokens=180
    )

    if 50 <= len(story.split()) <= 100:
        return story

    story = generate_story_once(
        build_retry_prompt(caption, mode, voice_style),
        max_new_tokens=180
    )

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."

    return story

# =========================
# AUDIO HELPERS
# =========================

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

# =========================
# RESET HELPERS
# =========================

def partial_reset_story_audio():
    st.session_state.last_story = None
    st.session_state.last_audio_bytes = None

def full_reset_app():
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_image_name = None
    st.session_state.last_caption = None
    st.session_state.last_story = None
    st.session_state.last_audio_bytes = None
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
# STORE UPLOADED IMAGE
# =========================

if uploaded_image is not None:
    uploaded_bytes = uploaded_image.getvalue()

    if (
        st.session_state.uploaded_image_name != uploaded_image.name
        or st.session_state.uploaded_image_bytes != uploaded_bytes
    ):
        st.session_state.uploaded_image_bytes = uploaded_bytes
        st.session_state.uploaded_image_name = uploaded_image.name
        st.session_state.last_caption = None
        st.session_state.last_story = None
        st.session_state.last_audio_bytes = None

# =========================
# DISPLAY IMAGE
# =========================

if st.session_state.uploaded_image_bytes is not None:
    image = load_image_from_bytes(st.session_state.uploaded_image_bytes)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.session_state.last_caption is None:
        with st.spinner("Generating caption..."):
            st.session_state.last_caption = img2text(image)

    if st.button("Generate Story & Audio", type="primary"):
        try:
            caption = st.session_state.last_caption

            with st.spinner("Writing cheerful story..."):
                story = text2story(caption, story_mode, voice_style)

            with st.spinner("Creating voice audio..."):
                voice_audio, sr = generate_voice_audio(story, voice_style, voice_tone)
                audio_bytes = audio_to_wav_bytes(voice_audio, sr)

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

# =========================
# RESET BUTTONS
# =========================

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Story & Audio"):
        partial_reset_story_audio()
        st.rerun()

with col2:
    if st.button("Reset App"):
        full_reset_app()

if st.session_state.uploaded_image_bytes is None:
    st.info("Please upload an image to begin.")
