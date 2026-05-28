import io
import random
import re
import wave
from collections import Counter

import cv2
import librosa
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
    div[data-testid="stAlert"],
    div[data-testid="stInfo"] {
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
        "last_base_caption": None,
        "last_face_summary": None,
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
    st.session_state.last_base_caption = None
    st.session_state.last_face_summary = None
    st.session_state.last_story = None
    st.session_state.last_audio_bytes = None
    st.session_state.reset_counter += 1
    st.rerun()


# ---------- Device helpers ----------

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pipeline_device():
    return 0 if torch.cuda.is_available() else -1


# ---------- Models ----------

@st.cache_resource
def get_img2text_model():
    try:
        return pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=get_pipeline_device()
        )
    except Exception:
        return None


@st.cache_resource
def get_expression_model():
    return pipeline(
        "image-classification",
        model="mo-thecreator/vit-Facial-Expression-Recognition",
        device=get_pipeline_device()
    )


@st.cache_resource
def get_tts_model():
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device=get_pipeline_device()
    )


@st.cache_resource
def get_story_components():
    model_name = "gpt2-medium"
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"low_cpu_mem_usage": True}

    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    return tokenizer, model


@st.cache_resource
def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise ValueError("Failed to load OpenCV Haar cascade for face detection.")
    return detector


# ---------- Image ----------

def validate_uploaded_image(uploaded_image):
    if uploaded_image is None:
        return

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    name = uploaded_image.name.lower()

    if not any(name.endswith(ext) for ext in allowed_exts):
        raise ValueError("Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP.")


def load_image_from_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("This image file could not be identified. Please upload a valid JPG, JPEG, PNG, or WEBP image.")
    except Exception as e:
        raise ValueError(
            f"Unable to read this image file. Please upload a valid JPG, JPEG, PNG, or WEBP image. Details: {e}"
        )


def save_uploaded_image(uploaded_image):
    if uploaded_image is None:
        return

    validate_uploaded_image(uploaded_image)
    uploaded_bytes = uploaded_image.getvalue()

    if (
        st.session_state.uploaded_image_name != uploaded_image.name
        or st.session_state.uploaded_image_bytes != uploaded_bytes
    ):
        st.session_state.uploaded_image_bytes = uploaded_bytes
        st.session_state.uploaded_image_name = uploaded_image.name
        st.session_state.last_caption = None
        st.session_state.last_base_caption = None
        st.session_state.last_face_summary = None
        st.session_state.last_story = None
        st.session_state.last_audio_bytes = None


def pil_to_cv2(image):
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_faces(image):
    detector = get_face_detector()
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def crop_faces(image, faces):
    rgb = np.array(image)
    crops = []

    for (x, y, w, h) in faces:
        face_crop = rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
        crops.append(Image.fromarray(face_crop))
    return crops


def normalize_expression_label(label):
    label = str(label).strip().lower()
    mapping = {
        "happiness": "happy",
        "happy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "sad": "sad",
        "anger": "angry",
        "angry": "angry",
        "surprise": "surprised",
        "fear": "fearful",
        "disgust": "disgusted",
        "excited": "excited"
    }
    return mapping.get(label, label)


def detect_facial_expressions(face_crops):
    if not face_crops:
        return []

    expr_model = get_expression_model()
    expressions = []

    for crop in face_crops:
        try:
            preds = expr_model(crop)
            if preds:
                top_label = preds[0]["label"]
                expressions.append(normalize_expression_label(top_label))
        except Exception:
            continue

    return expressions


def build_face_expression_summary(faces, expressions):
    face_count = len(faces)

    if face_count == 0:
        return ""

    if not expressions:
        return f"{face_count} face(s) detected, but the expressions were not clear enough to identify."

    counts = Counter(expressions)
    parts = [f"{emotion} ({count})" for emotion, count in counts.most_common()]
    joined = ", ".join(parts)

    if face_count == 1:
        return f"1 face detected. The expression appears {expressions[0]}."
    return f"{face_count} faces detected. The expressions appear to be: {joined}."


def get_emotion_words(expressions):
    if not expressions:
        return ""

    counts = Counter(expressions)
    emotion_list = [emotion for emotion, _ in counts.most_common()]
    return ", ".join(emotion_list)


def image_to_caption(image):
    model = get_img2text_model()

    if model is None:
        return "a happy family enjoying a special outdoor moment together"

    try:
        output = model(image)

        if not output:
            return "a happy family enjoying a special outdoor moment together"

        first = output[0]
        caption = (
            first.get("generated_text")
            or first.get("text")
            or ""
        ).strip()

        if not caption:
            return "a happy family enjoying a special outdoor moment together"

        return caption

    except Exception:
        return "a happy family enjoying a special outdoor moment together"


def image_to_caption_with_expression(image):
    base_caption = image_to_caption(image)
    faces = detect_faces(image)
    face_crops = crop_faces(image, faces)
    expressions = detect_facial_expressions(face_crops)
    face_summary = build_face_expression_summary(faces, expressions)
    emotion_words = get_emotion_words(expressions)

    if emotion_words:
        enriched_caption = f"{base_caption}. The people look {emotion_words}."
    else:
        enriched_caption = base_caption

    return base_caption, face_summary, enriched_caption, emotion_words


# ---------- Story safety ----------

def contains_unsafe_kids_content(text):
    unsafe_terms = [
        "kill", "killed", "killing", "murder", "blood", "bloody", "knife", "gun",
        "dead", "death", "die", "dying", "attack", "attacked", "violence", "violent",
        "hate", "hated", "naked", "sexy", "sex", "abuse", "abused", "drug", "drugs",
        "alcohol", "beer", "wine", "terror", "monster killed", "fight to death"
    ]

    lowered = text.lower()
    return any(term in lowered for term in unsafe_terms)


def safe_story_fallback(emotion_words=""):
    if emotion_words:
        return (
            f"One bright day, a happy little family shared a lovely moment together. "
            f"They looked {emotion_words}, and their smiles made the day feel extra special. "
            f"They talked, laughed, and stayed close to one another. "
            f"Everything felt warm, gentle, and full of care, and the day ended with a cozy, happy feeling."
        )
    return (
        "One bright day, a happy little family shared a lovely moment together. "
        "They smiled, talked, and enjoyed being close to one another. "
        "Everything felt warm, gentle, and full of care. "
        "Soon, everyone felt cheerful and safe, and the day ended with a cozy, happy feeling."
    )


def enforce_kid_safe_story(story, emotion_words=""):
    if not story or len(story.split()) < 20:
        return safe_story_fallback(emotion_words)

    if contains_unsafe_kids_content(story):
        return safe_story_fallback(emotion_words)

    return story


# ---------- Story cleaning ----------

def trim_to_last_complete_sentence(text):
    matches = list(re.finditer(r"[.!?]", text))
    if not matches:
        return text.strip()

    last_end = matches[-1].end()
    return text[:last_end].strip()


def remove_unwanted_patterns(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\b\d+%\b", "", text)
    text = re.sub(r"\b\d+\s*percent\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_emotions_in_story(story, emotion_words):
    if not emotion_words:
        return story

    if emotion_words.lower() not in story.lower():
        story += f" Everyone looked {emotion_words}, and that made the moment feel even more special."

    return story


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


def extract_story_only(generated_text, prompt):
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    markers = [
        "Picture description:",
        "Requirements:",
        "Story version:",
        "Story:"
    ]
    for marker in markers:
        if marker in generated_text:
            generated_text = generated_text.split(marker)[0]

    return generated_text.strip()


def clean_story(text, emotion_words=""):
    banned_phrases = [
        "story mode", "selected mode", "picture description", "image caption",
        "caption:", "prompt", "instruction", "story version", "rewrite version",
        "new story version", "rules:", "requirements:"
    ]

    text = text.replace("\\n", " ").replace("\n", " ").strip()
    text = remove_unwanted_patterns(text)
    text = re.sub(r"\s+", " ", text)

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
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
        if len(sentence.split()) < 3:
            continue

        cleaned.append(sentence)
        seen.add(lower_sentence)

    story = " ".join(cleaned).strip()
    story = trim_to_last_complete_sentence(story)

    if story and story[-1] not in ".!?":
        story += "."

    story = ensure_emotions_in_story(story, emotion_words)
    story = trim_to_last_complete_sentence(story)

    words = story.split()
    if len(words) > 100:
        shortened = " ".join(words[:100]).rstrip(".,;:! ")
        story = trim_to_last_complete_sentence(shortened)
        if not story:
            story = shortened + "."

    return story


def build_story_prompt(caption, mode, voice_style, emotion_words=""):
    story_seed = random.randint(1000, 999999)

    extra_emotion_line = ""
    if emotion_words:
        extra_emotion_line = (
            f"The story must clearly mention that the people look {emotion_words}. "
            f"Repeat these feelings naturally in the story."
        )

    return f"""
Write a short, friendly children's story of about 60 to 90 words.

Use the picture description carefully.
Keep the story cheerful, easy to understand, imaginative, complete, and suitable for young kids.
Do not include violence, scary content, romance for adults, harmful behavior, references, citations, brackets, scores, percentages, or statistics.
Every sentence must be complete and natural for children.
{extra_emotion_line}
{get_style_instruction(mode)}
{get_voice_instruction(voice_style)}

Picture description: {caption}

Story version: {story_seed}

Story:
""".strip()


def generate_story_once(prompt, emotion_words="", max_new_tokens=140):
    tokenizer, model = get_story_components()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = extract_story_only(text, prompt)
    story = clean_story(story, emotion_words)
    story = enforce_kid_safe_story(story, emotion_words)
    return story


def caption_to_story(caption, mode, voice_style, emotion_words=""):
    prompt = build_story_prompt(caption, mode, voice_style, emotion_words)
    best_story = ""

    for _ in range(3):
        story = generate_story_once(prompt, emotion_words=emotion_words)
        wc = len(story.split())
        best_story = story

        if (
            50 <= wc <= 100
            and not contains_unsafe_kids_content(story)
            and "["
            not in story
            and "%"
            not in story
        ):
            return story

    if len(best_story.split()) < 50:
        if emotion_words:
            best_story += f" Everyone looked {emotion_words}, and their happy feelings made the day shine brightly."
        else:
            best_story += " Everyone smiled, and the day felt warm, bright, and full of love."

    best_story = clean_story(best_story, emotion_words)
    return enforce_kid_safe_story(best_story, emotion_words)


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
    if st.session_state.last_base_caption:
        st.success("🐻 The parent bears looked at your picture.")
        st.write(f"**What the bears saw:** {st.session_state.last_base_caption}")

    if st.session_state.last_face_summary:
        st.info(f"**Faces and expressions:** {st.session_state.last_face_summary}")

    if st.session_state.last_story:
        st.markdown("### 🐻 Your little-bear story")
        st.write(st.session_state.last_story)

    if st.session_state.last_audio_bytes:
        st.success("🐻 The bear voice is ready!")
        st.audio(st.session_state.last_audio_bytes, format="audio/wav")
        st.download_button(
            "🐻 Download the bear voice",
            data=st.session_state.last_audio_bytes,
            file_name="kids_story.wav",
            mime="audio/wav"
        )

    if st.session_state.last_story or st.session_state.last_audio_bytes:
        if st.button("🐻 Make another story with a new image"):
            reset_for_another_story()


def generate_story_and_audio(image, story_mode, voice_style, voice_tone):
    if not st.session_state.last_caption:
        with st.spinner("📸 A tiny owl is peeking at your picture..."):
            base_caption, face_summary, enriched_caption, emotion_words = image_to_caption_with_expression(image)
            st.session_state.last_base_caption = base_caption
            st.session_state.last_face_summary = face_summary
            st.session_state.last_caption = enriched_caption
            st.session_state.last_emotion_words = emotion_words
    else:
        emotion_words = st.session_state.get("last_emotion_words", "")

    caption = st.session_state.last_caption

    with st.spinner("🍰 Your story is in the oven!"):
        story = caption_to_story(caption, story_mode, voice_style, emotion_words)

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
    "Upload an image and let the parent bears turn it into a child-friendly story and voice."
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
    type=["jpg", "jpeg", "png", "webp"],
    key=f"uploader_{suffix}"
)

try:
    save_uploaded_image(uploaded_image)
except Exception as e:
    st.error(str(e))
    uploaded_image = None

if st.session_state.uploaded_image_bytes is not None:
    try:
        image = load_image_from_bytes(st.session_state.uploaded_image_bytes)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("🐻 Ask the parent bears to tell a story", type="primary"):
            try:
                generate_story_and_audio(image, story_mode, voice_style, voice_tone)
            except RuntimeError as e:
                st.error(f"Memory or model loading issue: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    except Exception as e:
        st.error(str(e))

show_results()

if st.session_state.uploaded_image_bytes is None:
    st.info("Please upload an image to begin.")
