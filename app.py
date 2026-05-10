import io
import random
import wave

import librosa
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline


APP_TITLE = "🐻🐾 Parent Bears are telling stories right now! Let's join with other little bears! 🐾🐻"


# ---------- Style ----------

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(255, 255, 255, 0.70) 0%, rgba(255, 255, 255, 0) 18%),
            radial-gradient(circle at 85% 15%, rgba(255, 244, 189, 0.65) 0%, rgba(255, 244, 189, 0) 20%),
            radial-gradient(circle at 20% 85%, rgba(255, 214, 232, 0.55) 0%, rgba(255, 214, 232, 0) 22%),
            radial-gradient(circle at 80% 80%, rgba(189, 234, 255, 0.55) 0%, rgba(189, 234, 255, 0) 20%),
            linear-gradient(180deg, #fff9e8 0%, #ffeef6 45%, #eef9ff 100%);
        background-attachment: fixed;
    }

    .stApp > header {
        background: rgba(255, 255, 255, 0);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stFileUploader"]) {
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid rgba(255, 193, 214, 0.65);
        border-radius: 22px;
        padding: 1rem 1rem 1.2rem 1rem;
        box-shadow: 0 10px 30px rgba(201, 157, 123, 0.10);
    }

    div[data-testid="stButton"] > button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #ff8fab 0%, #ff758f 100%);
        color: white;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
        box-shadow: 0 8px 18px rgba(255, 117, 143, 0.25);
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #ff7998 0%, #ff5c7c 100%);
        color: white;
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #7cc7ff 0%, #5caeff 100%);
        color: white;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
        box-shadow: 0 8px 18px rgba(92, 174, 255, 0.25);
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #64baff 0%, #439eff 100%);
        color: white;
    }

    div[data-testid="stSuccess"],
    div[data-testid="stAlert"] {
        border-radius: 18px;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------- State ----------

def init_state():
    defaults = {
        "entry_confirmed": None,
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


def reset_for_another_story():
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_image_name = None
    st.session_state.last_caption = None
    st.session_state.last_story = None
    st.session_state.last_audio_bytes = None
    st.session_state.reset_counter += 1
    st.rerun()


# ---------- Models ----------

@st.cache_resource
def get_img2text_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=device
    )


@st.cache_resource
def get_story_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=device
    )


@st.cache_resource
def get_tts_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device=device
    )


# ---------- Image ----------

def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def save_uploaded_image(uploaded_image):
    if uploaded_image is None:
        return

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


def image_to_caption(image):
    model = get_img2text_model()
    output = model(image)
    return output[0].get("generated_text", "").strip()


# ---------- Story ----------

def get_style_instruction(mode):
    style_map = {
        "Fairy-tale": "Write it as a magical fairy tale with wonder, charm, and a happy ending.",
        "Adventure": "Write it as a playful adventure with discovery and excitement.",
        "Bedtime": "Write it as a calm bedtime story with a peaceful and cozy feeling.",
        "Silly / Funny": "Write it as a silly and funny story that may make kids laugh.",
        "Superhero": "Write it as a gentle superhero story with courage, kindness, and fun."
    }
    return style_map.get(mode, "Write it as a cheerful children's story.")


def get_voice_instruction(voice_style):
    voice_map = {
        "Friendly narrator": "Use a warm and friendly storytelling tone.",
        "Soft bedtime voice": "Use a soft and gentle storytelling tone.",
        "Excited storyteller": "Use an energetic and lively storytelling tone.",
        "Cartoonish voice": "Use a playful and cartoon-like storytelling tone."
    }
    return voice_map.get(voice_style, "Use a cheerful storytelling tone.")


def clean_story(text):
    banned_phrases = [
        "story mode", "selected mode", "picture description", "image caption",
        "caption:", "prompt", "instruction", "story version", "rewrite version",
        "new story version", "the story mode is", "write one brand new",
        "write a fresh", "rules:", "requirements:"
    ]

    text = text.replace("\n", " ").strip()
    raw_sentences = text.split(".")
    cleaned = []
    seen = set()

    for sentence in raw_sentences:
        sentence = " ".join(sentence.strip().split())
        lower_sentence = sentence.lower()

        if not sentence:
            continue
        if any(phrase in lower_sentence for phrase in banned_phrases):
            continue
        if lower_sentence in seen:
            continue

        cleaned.append(sentence)
        seen.add(lower_sentence)

    story = ". ".join(cleaned).strip()

    if story and not story.endswith("."):
        story += "."

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."

    return story


def generate_story_once(prompt, max_new_tokens=180):
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


def build_main_story_prompt(caption, mode, voice_style):
    story_seed = random.randint(1000, 999999)
    return f"""
Picture description: {caption}

Write one brand new children's story in 50 to 100 words.

Requirements:
- Use the main nouns, objects, animals, and actions from the picture description.
- Keep the story clearly related to the picture description.
- {get_style_instruction(mode)}
- {get_voice_instruction(voice_style)}
- Make it cheerful, imaginative, playful, and suitable for young kids.
- Use simple English.
- Do not repeat sentences.
- Do not mention labels like picture description, caption, prompt, instruction, or story mode.
- End with a happy or warm feeling.

Story version: {story_seed}

Story:
""".strip()


def build_expand_prompt(caption, mode, voice_style, short_story):
    rewrite_seed = random.randint(1000, 999999)
    return f"""
Picture description: {caption}

Here is a short children's story:
{short_story}

Rewrite and expand it into one complete children's story in 50 to 100 words.

Requirements:
- Keep it clearly related to the picture description.
- Keep the same core meaning, but add more cheerful and imaginative details.
- {get_style_instruction(mode)}
- {get_voice_instruction(voice_style)}
- Use simple English for kids.
- Do not repeat sentences.
- Do not mention prompts, labels, caption, or story mode.
- End with a happy or warm feeling.

Rewrite version: {rewrite_seed}

Improved story:
""".strip()


def build_retry_prompt(caption, mode, voice_style):
    retry_seed = random.randint(1000, 999999)
    return f"""
Picture description: {caption}

Write a completely new children's story in 50 to 100 words.

Requirements:
- The story must be clearly based on the picture description.
- Use the important nouns and actions from the picture description.
- {get_style_instruction(mode)}
- {get_voice_instruction(voice_style)}
- Make it cheerful, imaginative, playful, and suitable for kids.
- Use simple English.
- Do not copy the picture description as a single sentence.
- Do not repeat sentences.
- Do not mention prompts, labels, caption, or story mode.
- End happily.

New story version: {retry_seed}

Story:
""".strip()


def caption_to_story(caption, mode, voice_style):
    story = generate_story_once(build_main_story_prompt(caption, mode, voice_style))

    if 50 <= len(story.split()) <= 100:
        return story

    story = generate_story_once(build_expand_prompt(caption, mode, voice_style, story))

    if 50 <= len(story.split()) <= 100:
        return story

    story = generate_story_once(build_retry_prompt(caption, mode, voice_style))

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."

    return story


# ---------- Audio ----------

def apply_voice_effects(audio, sr, voice_style, voice_tone):
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

    if voice_tone == "Old Bear":
        pitch_steps -= 2
    elif voice_tone == "Teenage Bear":
        pitch_steps += 2

    if pitch_steps != 0:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)

    return np.clip(audio, -1.0, 1.0)


def text_to_audio(text, voice_style, voice_tone):
    tts = get_tts_model()
    output = tts(text)

    if "audio" not in output or "sampling_rate" not in output:
        raise ValueError("TTS model did not return valid audio output.")

    audio = np.asarray(output["audio"], dtype=np.float32).squeeze()
    sr = int(output["sampling_rate"])
    audio = apply_voice_effects(audio, sr, voice_style, voice_tone)

    return audio, sr


def audio_to_wav_bytes(audio, sr):
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


# ---------- UI ----------

def show_entry_gate():
    if st.session_state.entry_confirmed is None:
        st.title("🐻 Welcome, Little Bear 🐾")
        st.warning(
            "This website only generates stories for children under 18. "
            "Do you understand and wish to enter?"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Yes"):
                st.session_state.entry_confirmed = True
                st.rerun()

        with col2:
            if st.button("No"):
                st.session_state.entry_confirmed = False
                st.rerun()

        st.stop()

    if st.session_state.entry_confirmed is False:
        st.title("🐻 Access Notice")
        st.error("Please leave this website.")
        st.stop()


def show_results():
    if st.session_state.last_caption:
        st.success("🐻 Parent bears vote a theme of a story!")
        st.write(f"**Story theme:** {st.session_state.last_caption}")

    if st.session_state.last_story:
        st.markdown("### 🐻 Parent bears are telling story now....")
        st.write(st.session_state.last_story)

    if st.session_state.last_audio_bytes:
        st.success("🐻 Voice of the roarrrrr !")
        st.audio(st.session_state.last_audio_bytes, format="audio/wav")
        st.download_button(
            "🐻 Download the voice of bear",
            data=st.session_state.last_audio_bytes,
            file_name="kids_story.wav",
            mime="audio/wav"
        )

    if st.session_state.last_story or st.session_state.last_audio_bytes:
        if st.button("🐻 Discuss with bears for another story with a new image!"):
            reset_for_another_story()


def generate_story_and_audio(image, story_mode, voice_style, voice_tone):
    if not st.session_state.last_caption:
        with st.spinner("📸 A tiny owl is peeking at your picture..."):
            st.session_state.last_caption = image_to_caption(image)

    caption = st.session_state.last_caption

    with st.spinner("🍰 Your story is in the oven!"):
        story = caption_to_story(caption, story_mode, voice_style)

    with st.spinner("🐻 Some little bears are tasting your story!"):
        voice_audio, sr = text_to_audio(story, voice_style, voice_tone)
        audio_bytes = audio_to_wav_bytes(voice_audio, sr)

    st.session_state.last_story = story
    st.session_state.last_audio_bytes = audio_bytes


# ---------- Main ----------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐻",
    layout="centered"
)

apply_custom_css()
init_state()
show_entry_gate()

st.title(APP_TITLE)
st.write(
    "Upload an image, let the parent bears look at it, and enjoy a story and audio made for little bears."
)

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
    "Choose a bear voice",
    ["Adult Bear", "Old Bear", "Teenage Bear"],
    index=0,
    key=f"voice_tone_{suffix}"
)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{suffix}"
)

save_uploaded_image(uploaded_image)

if st.session_state.uploaded_image_bytes is not None:
    image = load_image_from_bytes(st.session_state.uploaded_image_bytes)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🐻 Discuss with parent bears for a story", type="primary"):
        try:
            generate_story_and_audio(image, story_mode, voice_style, voice_tone)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

show_results()

if st.session_state.uploaded_image_bytes is None:
    st.info("Please upload an image to begin.")
