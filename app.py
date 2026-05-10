import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import tempfile
import wave
import librosa

# =========================
# MAIN UI
# =========================

st.title("📖 Cheerful Kids' Image Storytelling App")

st.write(
    "Upload an image → get a caption → choose a story mode and voice → "
    "generate a magical kids story with audio."
)

# =========================
# MODELS
# =========================

@st.cache_resource
def get_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def get_story_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def get_tts_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")


# =========================
# HELPERS
# =========================

def img2text(image):
    model = get_img2text_model()
    return model(image)[0]["generated_text"]


def build_story_prompt(caption, mode):
    base = (
        f"Image caption: {caption}\n\n"
        "Write a 50–100 word children's story based on this caption. "
        "The story must stay consistent with the caption, but you may add gentle, imaginative background details "
        "that are not shown in the picture (like warm sunshine, sparkles, soft magic, or friendly sounds). "
        "Do NOT repeat sentences. Do NOT loop phrases. "
        "Make the story cheerful, magical, and easy for young kids.\n\n"
    )

    if mode == "Fairy-tale":
        style = "Write it in a soft fairy-tale style with friendly magic and a cozy ending.\n\n"
    elif mode == "Adventure":
        style = "Write it in a light adventure style with gentle excitement.\n\n"
    elif mode == "Bedtime":
        style = "Write it in a calm bedtime style with soothing words.\n\n"
    elif mode == "Silly / Funny":
        style = "Write it in a silly, funny style with playful humor.\n\n"
    elif mode == "Superhero":
        style = "Write it in a gentle superhero style, brave but friendly.\n\n"
    else:
        style = ""

    structure = (
        "Structure:\n"
        "- 1 sentence describing the scene\n"
        "- 2–3 sentences adding magical or cheerful background\n"
        "- 1 sentence ending with a warm feeling\n\n"
        "Story:"
    )

    return base + style + structure


def text2story(caption, mode):
    model = get_story_model()
    prompt = build_story_prompt(caption, mode)

    output = model(
        prompt,
        max_new_tokens=180,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=3.5
    )[0]["generated_text"]

    story = output.strip()

    # Remove repeated sentences
    sentences = story.split(".")
    cleaned = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if len(s) > 0 and s not in seen:
            cleaned.append(s)
            seen.add(s)

    story = ". ".join(cleaned).strip() + "."
    words = story.split()

    # Enforce ~50–100 words
    if len(words) < 50:
        story += " Soft sparkles of magic drifted gently through the air, making everything feel warm and full of wonder."
    elif len(words) > 100:
        story = " ".join(words[:100]) + "."

    return story


def generate_voice_audio(text, voice_style, gender):
    tts = get_tts_model()
    out = tts(text)

    # Convert list → numpy array for librosa
    audio = np.array(out["audio"]).astype(np.float32)
    sr = out["sampling_rate"]

    # Apply gender-based pitch shift
    if gender == "Male":
        audio = librosa.effects.pitch_shift(audio, sr, n_steps=-3)
    elif gender == "Female":
        audio = librosa.effects.pitch_shift(audio, sr, n_steps=3)

    return audio, sr


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
# SESSION STATE
# =========================

if "last_story" not in st.session_state:
    st.session_state.last_story = None
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None


# =========================
# CONTROLS
# =========================

story_mode = st.selectbox(
    "Choose a story mode",
    ["Fairy-tale", "Adventure", "Bedtime", "Silly / Funny", "Superhero"]
)

voice_style = st.selectbox(
    "Choose a voice style",
    ["Friendly narrator", "Soft bedtime voice", "Excited storyteller", "Cartoonish voice"]
)

gender = st.selectbox(
    "Choose voice gender",
    ["Male", "Female"]
)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# =========================
# MAIN LOGIC
# =========================

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Story & Audio"):
        with st.spinner("Generating caption..."):
            caption = img2text(image)
        st.success("Caption generated!")
        st.write(f"**Caption:** {caption}")

        with st.spinner("Writing cheerful story..."):
            story = text2story(caption, story_mode)
        st.success("Story created!")
        st.write("### 📘 Your Story")
        st.write(story)

        with st.spinner("Creating voice audio..."):
            voice_audio, sr = generate_voice_audio(story, voice_style, gender)

        audio_path = save_audio(voice_audio, sr)
        st.success("Audio ready!")
        st.audio(audio_path)

        st.session_state.last_story = story
        st.session_state.last_audio_path = audio_path

    if st.session_state.last_audio_path and st.button("🔁 Read Again"):
        st.write("### 📘 Your Story")
        st.write(st.session_state.last_story)
        st.audio(st.session_state.last_audio_path)

    if st.button("Reset App"):
        st.session_state.last_story = None
        st.session_state.last_audio_path = None
        st.experimental_rerun()
