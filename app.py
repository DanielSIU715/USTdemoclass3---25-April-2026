import io
import json
import re
import wave
import random
from collections import Counter
from pathlib import Path

import cv2
import librosa
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# =========================================================
# CONSTANTS
# =========================================================

APP_TITLE = "🐻🐾 Parent Bears are telling stories right now! Let's join with other little bears! 🐾🐻"
DEFAULT_CAPTION = "a happy family enjoying a special outdoor moment together"
DEFAULT_EMOTION = "happy"

DEFAULT_USERNAME = "Little UST Bear"
DEFAULT_PASSWORD = "123456"

SCORES_FILE = "user_scores.json"

ANIMAL_INFO = {
    "Bear": {"icon": "🐻"},
    "Owl": {"icon": "🦉"},
    "Dog": {"icon": "🐶"},
    "Rabbit": {"icon": "🐰"},
    "Fox": {"icon": "🦊"},
    "Panda": {"icon": "🐼"},
}


# =========================================================
# FUNCTIONS
# =========================================================

# ---------- Styling functions ----------

def apply_custom_css():
    """Apply the visual style for the app interface."""
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


# ---------- Score storage functions ----------

def default_animal_scores():
    """Create the initial animal subtotal dictionary."""
    return {animal: 0 for animal in ANIMAL_INFO.keys()}


def load_score_data():
    """Load saved user score data from the local JSON file."""
    path = Path(SCORES_FILE)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_score_data(data):
    """Save all user score data to the local JSON file."""
    path = Path(SCORES_FILE)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_default_user():
    """Ensure the default demo user exists in the score file."""
    data = load_score_data()
    if DEFAULT_USERNAME not in data:
        data[DEFAULT_USERNAME] = {
            "password": DEFAULT_PASSWORD,
            "total_score": 0,
            "score_history": [],
            "animal_scores": default_animal_scores(),
        }
        save_score_data(data)
        return

    changed = False
    if "animal_scores" not in data[DEFAULT_USERNAME]:
        data[DEFAULT_USERNAME]["animal_scores"] = default_animal_scores()
        changed = True

    for animal in ANIMAL_INFO.keys():
        if animal not in data[DEFAULT_USERNAME]["animal_scores"]:
            data[DEFAULT_USERNAME]["animal_scores"][animal] = 0
            changed = True

    if changed:
        save_score_data(data)


def get_user_data(username):
    """Get one user's score record, including total and animal subtotals."""
    data = load_score_data()
    user_data = data.get(username)

    if not user_data:
        user_data = {
            "password": DEFAULT_PASSWORD,
            "total_score": 0,
            "score_history": [],
            "animal_scores": default_animal_scores(),
        }

    if "animal_scores" not in user_data:
        user_data["animal_scores"] = default_animal_scores()

    for animal in ANIMAL_INFO.keys():
        user_data["animal_scores"].setdefault(animal, 0)

    user_data.setdefault("total_score", 0)
    user_data.setdefault("score_history", [])

    return user_data


def get_total_score(username):
    """Return the total number of animals collected by the user."""
    user_data = get_user_data(username)
    return int(user_data.get("total_score", 0))


def get_animal_scores(username):
    """Return the subtotal score for each animal type."""
    user_data = get_user_data(username)
    return user_data.get("animal_scores", default_animal_scores())


def choose_animal_for_score():
    """Randomly choose which animal receives the current story score."""
    return random.choice(list(ANIMAL_INFO.keys()))


def add_score_to_user(username, score, animal_name):
    """Add the new story score to the user's total and one animal subtotal."""
    data = load_score_data()

    if username not in data:
        data[username] = {
            "password": DEFAULT_PASSWORD,
            "total_score": 0,
            "score_history": [],
            "animal_scores": default_animal_scores(),
        }

    data[username].setdefault("animal_scores", default_animal_scores())
    for animal in ANIMAL_INFO.keys():
        data[username]["animal_scores"].setdefault(animal, 0)

    data[username]["total_score"] = int(data[username].get("total_score", 0)) + int(score)
    data[username]["score_history"].append(int(score))
    data[username]["animal_scores"][animal_name] = int(data[username]["animal_scores"].get(animal_name, 0)) + int(score)

    save_score_data(data)


# ---------- Session state functions ----------

def init_state():
    """Initialize all session-state variables used by the app."""
    defaults = {
        "entry_confirmed": None,
        "logged_in": False,
        "current_user": "",
        "uploaded_image_bytes": None,
        "uploaded_image_name": None,
        "last_caption": None,
        "last_base_caption": None,
        "last_face_summary": None,
        "last_story": None,
        "last_audio_bytes": None,
        "last_emotion_words": DEFAULT_EMOTION,
        "kid_score": None,
        "kid_score_message": None,
        "score_saved_for_current_story": False,
        "last_score_animal": None,
        "reset_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def widget_key(name):
    """Create a reset-safe widget key."""
    return f"{name}_{st.session_state.reset_counter}"


def clear_generated_outputs():
    """Clear generated story, audio, and scoring results for a fresh story cycle."""
    st.session_state.last_caption = None
    st.session_state.last_base_caption = None
    st.session_state.last_face_summary = None
    st.session_state.last_story = None
    st.session_state.last_audio_bytes = None
    st.session_state.last_emotion_words = DEFAULT_EMOTION
    st.session_state.kid_score = None
    st.session_state.kid_score_message = None
    st.session_state.score_saved_for_current_story = False
    st.session_state.last_score_animal = None


def reset_for_another_story():
    """Reset only the story-generation workflow while keeping the user logged in."""
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_image_name = None
    clear_generated_outputs()
    st.session_state.reset_counter += 1
    st.rerun()


def logout_and_return_to_warning():
    """Log out the user and return the app to the original warning page."""
    keys_to_keep = []
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.rerun()


# ---------- Device and model helper functions ----------

def get_device():
    """Return the best available torch device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pipeline_device():
    """Return Streamlit pipeline device index for Hugging Face pipelines."""
    return 0 if torch.cuda.is_available() else -1


@st.cache_resource
def get_img2text_model():
    """Load the image captioning model."""
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
    """Load the facial expression recognition model."""
    return pipeline(
        "image-classification",
        model="mo-thecreator/vit-Facial-Expression-Recognition",
        device=get_pipeline_device()
    )


@st.cache_resource
def get_tts_model():
    """Load the text-to-speech model."""
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device=get_pipeline_device()
    )


@st.cache_resource
def get_story_components():
    """Load the tokenizer and story-generation model."""
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
    """Load the OpenCV face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise ValueError("Failed to load OpenCV Haar cascade for face detection.")
    return detector


# ---------- General helper functions ----------

def get_current_form_values():
    """Read all current child input fields from the screen."""
    return {
        "background": st.session_state.get(widget_key("kid_background"), "").strip(),
        "about": st.session_state.get(widget_key("kid_photo_about"), "").strip(),
        "doing": st.session_state.get(widget_key("kid_doing"), "").strip(),
        "special": st.session_state.get(widget_key("kid_special"), "").strip(),
        "where": st.session_state.get(widget_key("kid_where"), "").strip(),
    }


def simplify_text(text):
    """Replace harder words with simpler child-friendly wording."""
    replacements = {
        "shared a lovely time together": "had a nice time together",
        "special moment": "happy moment",
        "lovely memory": "happy memory",
        "extra warm and happy": "very warm and happy",
        "treasure": "remember",
        "peaceful": "calm",
        "gentle": "soft",
        "enjoying": "having fun in",
        "togetherness": "love",
        "wonderful": "happy",
    }

    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)

    return out


def count_words(text):
    """Count the number of words in a story."""
    return len(re.findall(r"\b[\w']+\b", text))


def remove_html_and_code_fragments(text):
    """Remove HTML tags, comments, and code-like fragments from generated text."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^{}]*\}", " ", text)
    text = re.sub(r"\b[a-zA-Z_]+\s*=\s*['\"].*?['\"]", " ", text)
    text = re.sub(r"\b(width|height|style|class|id|src)\s*=\s*[^ ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:http|https)://\S+", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    return text


def remove_non_story_noise(text):
    """Remove suspicious code-related sentences that do not belong to a story."""
    noise_patterns = [
        r"iframe",
        r"fb_main",
        r"xmlns",
        r"javascript",
        r"function\s*\(",
        r"var\s+[a-zA-Z_]",
        r"document\.",
        r"window\.",
        r"<div",
        r"</div>",
        r"<span",
        r"</span>",
    ]

    cleaned_sentences = []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence_stripped = sentence.strip()
        if not sentence_stripped:
            continue

        lower_sentence = sentence_stripped.lower()
        if any(re.search(pattern, lower_sentence) for pattern in noise_patterns):
            continue

        cleaned_sentences.append(sentence_stripped)

    return " ".join(cleaned_sentences).strip()


# ---------- Image processing functions ----------

def validate_uploaded_image(uploaded_image):
    """Validate the uploaded image type."""
    if uploaded_image is None:
        return

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    name = uploaded_image.name.lower()

    if not any(name.endswith(ext) for ext in allowed_exts):
        raise ValueError("Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP.")


def load_image_from_bytes(image_bytes):
    """Convert uploaded bytes into a PIL image."""
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
    """Save a newly uploaded image into session state."""
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
        clear_generated_outputs()


def pil_to_cv2(image):
    """Convert a PIL image to an OpenCV image."""
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_faces(image):
    """Detect faces in the uploaded image."""
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
    """Crop each detected face for expression analysis."""
    rgb = np.array(image)
    crops = []

    for (x, y, w, h) in faces:
        face_crop = rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
        crops.append(Image.fromarray(face_crop))
    return crops


def normalize_expression_label(label):
    """Map model emotion labels into simpler child-friendly emotion words."""
    label = str(label).strip().lower()
    mapping = {
        "happiness": "happy",
        "happy": "happy",
        "neutral": "happy",
        "sadness": "sad",
        "sad": "sad",
        "anger": "angry",
        "angry": "angry",
        "surprise": "surprised",
        "fear": "scared",
        "disgust": "upset",
        "excited": "excited"
    }
    return mapping.get(label, DEFAULT_EMOTION)


def detect_facial_expressions(face_crops):
    """Predict facial emotions from detected face crops."""
    if not face_crops:
        return [DEFAULT_EMOTION]

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

    if not expressions:
        return [DEFAULT_EMOTION]

    return expressions


def build_face_expression_summary(faces, expressions):
    """Create a short summary of detected faces and their emotions."""
    if not expressions:
        expressions = [DEFAULT_EMOTION]

    face_count = len(faces)

    if face_count == 0:
        return f"No clear faces were detected, so the feeling is recorded as {DEFAULT_EMOTION}."

    counts = Counter(expressions)
    parts = [f"{emotion} ({count})" for emotion, count in counts.most_common()]
    joined = ", ".join(parts)

    if face_count == 1:
        return f"1 face detected. The expression appears {expressions[0]}."
    return f"{face_count} faces detected. The expressions appear to be: {joined}."


def get_emotion_words(expressions):
    """Turn detected emotions into a readable phrase."""
    if not expressions:
        return DEFAULT_EMOTION

    counts = Counter(expressions)
    emotion_list = [emotion for emotion, _ in counts.most_common()]

    if len(emotion_list) == 1:
        return emotion_list[0]
    if len(emotion_list) == 2:
        return f"{emotion_list[0]} and {emotion_list[1]}"
    return ", ".join(emotion_list[:-1]) + f", and {emotion_list[-1]}"


def image_to_caption(image):
    """Generate a basic caption for the uploaded image."""
    model = get_img2text_model()

    if model is None:
        return DEFAULT_CAPTION

    try:
        output = model(image)

        if not output:
            return DEFAULT_CAPTION

        first = output[0]
        caption = (first.get("generated_text") or first.get("text") or "").strip()

        if not caption:
            return DEFAULT_CAPTION

        return caption

    except Exception:
        return DEFAULT_CAPTION


def build_child_context(child_facts):
    """Combine the child's answers into one text block for story prompting."""
    parts = []

    if child_facts["background"]:
        parts.append(f"Background from the child: {child_facts['background']}")
    if child_facts["about"]:
        parts.append(f"What the photo is about: {child_facts['about']}")
    if child_facts["doing"]:
        parts.append(f"What the child was doing: {child_facts['doing']}")
    if child_facts["special"]:
        parts.append(f"Why the photo is special: {child_facts['special']}")
    if child_facts["where"]:
        parts.append(f"Where the photo was taken: {child_facts['where']}")

    return " ".join(parts).strip()


def image_to_caption_with_expression(image, child_facts):
    """Build an enriched caption using image content, child answers, and facial emotions."""
    base_caption = image_to_caption(image)
    faces = detect_faces(image)
    face_crops = crop_faces(image, faces)
    expressions = detect_facial_expressions(face_crops)
    face_summary = build_face_expression_summary(faces, expressions)
    emotion_words = get_emotion_words(expressions)
    child_context = build_child_context(child_facts)

    enriched_parts = [f"Observed photo caption: {base_caption}."]
    if child_context:
        enriched_parts.append(child_context)
    enriched_parts.append(f"The feeling in the photo is {emotion_words}.")

    enriched_caption = " ".join(part for part in enriched_parts if part).strip()

    return base_caption, face_summary, enriched_caption, emotion_words


# ---------- Story generation functions ----------

def contains_unsafe_kids_content(text):
    """Check whether a generated story contains unsafe words for children."""
    unsafe_terms = [
        "kill", "killed", "killing", "murder", "blood", "bloody", "knife", "gun",
        "dead", "death", "die", "dying", "attack", "attacked", "violence", "violent",
        "hate", "hated", "naked", "sexy", "sex", "abuse", "abused", "drug", "drugs",
        "alcohol", "beer", "wine", "terror", "fight to death"
    ]
    lowered = text.lower()
    return any(term in lowered for term in unsafe_terms)


def build_emotion_closing(emotion_words):
    """Create a warm closing sentence based on the detected emotions."""
    if not emotion_words:
        emotion_words = DEFAULT_EMOTION
    return f"At the end, all the bears looked {emotion_words}, and that made the day feel warm and happy."


def build_short_story_ending_fix(story, emotion_words):
    """Ensure the final sentence uses the required emotion-based ending."""
    sentences = re.split(r"(?<=[.!?])\s+", story.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    filtered = [s for s in sentences if "all the bears looked" not in s.lower()]
    filtered.append(build_emotion_closing(emotion_words))
    return " ".join(filtered)


def build_grounded_story(base_caption, emotion_words, child_facts):
    """Create a safe fallback story if the language model output is not usable."""
    where = child_facts["where"]
    about = child_facts["about"]
    doing = child_facts["doing"]
    special = child_facts["special"]
    background = child_facts["background"]

    sentences = []

    if where:
        sentences.append(f"One day at {where}, a little bear and the family had a nice time together.")
    else:
        sentences.append("One day, a little bear and the family had a nice time together.")

    sentences.append(f"The photo showed {base_caption}.")

    if about:
        sentences.append(f"This photo was about {about}.")
    if doing:
        sentences.append(f"In the picture, the little bear was {doing}.")
    if special:
        sentences.append(f"It was special because {special}.")
    if background:
        sentences.append(f"It was a happy memory because {background}.")

    sentences.append(build_emotion_closing(emotion_words))

    story = " ".join(sentences)
    story = simplify_text(story)

    words = story.split()
    if len(words) > 90:
        story = " ".join(words[:90]).rstrip(".,;:! ") + "."
        story = build_short_story_ending_fix(story, emotion_words)

    return story


def ensure_caption_in_story(story, base_caption):
    """Make sure the final story includes the key photo caption."""
    if base_caption.lower().strip() not in story.lower():
        story = f"The photo showed {base_caption}. " + story
    return story


def ensure_child_details_in_story(story, child_facts):
    """Make sure the final story contains the child's important answers."""
    story_lower = story.lower()

    if child_facts["about"] and child_facts["about"].lower() not in story_lower:
        story += f" This photo was about {child_facts['about']}."

    story_lower = story.lower()
    if child_facts["doing"] and child_facts["doing"].lower() not in story_lower:
        story += f" In the picture, the little bear was {child_facts['doing']}."

    story_lower = story.lower()
    if child_facts["where"] and child_facts["where"].lower() not in story_lower:
        story += f" This happy moment happened at {child_facts['where']}."

    story_lower = story.lower()
    if child_facts["special"] and child_facts["special"].lower() not in story_lower:
        story += f" It was special because {child_facts['special']}."

    return story


def ensure_emotion_closing(story, emotion_words):
    """Make sure the final story ends with the required warm emotion sentence."""
    return build_short_story_ending_fix(story, emotion_words or DEFAULT_EMOTION)


def clean_story(text, base_caption="", emotion_words="", child_facts=None):
    """Clean model output by removing prompt leakage, HTML/code fragments, and repeated text."""
    if child_facts is None:
        child_facts = {"background": "", "about": "", "doing": "", "special": "", "where": ""}

    text = text.replace("\\n", " ").replace("\n", " ").strip()
    text = remove_html_and_code_fragments(text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\s+", " ", text)

    banned_phrases = [
        "story mode", "selected mode", "picture description", "image caption",
        "caption:", "prompt", "instruction", "story version", "rewrite version",
        "new story version", "rules:", "requirements:", "observed photo caption:",
        "write a very simple", "child photo topic", "child action",
        "child special reason", "child place", "child background"
    ]

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
        if "<" in sentence or ">" in sentence:
            continue
        if "{" in sentence or "}" in sentence:
            continue
        if "style=" in lower_sentence or "width=" in lower_sentence or "height=" in lower_sentence:
            continue
        if lower_sentence in seen:
            continue
        if len(sentence.split()) < 3:
            continue

        cleaned.append(sentence)
        seen.add(lower_sentence)

    story = " ".join(cleaned).strip()
    story = remove_non_story_noise(story)
    story = re.sub(r"\s+", " ", story).strip()

    if story and story[-1] not in ".!?":
        story += "."

    story = ensure_caption_in_story(story, base_caption)
    story = ensure_child_details_in_story(story, child_facts)
    story = ensure_emotion_closing(story, emotion_words or DEFAULT_EMOTION)
    story = simplify_text(story)

    words = story.split()
    if len(words) > 100:
        story = " ".join(words[:100]).rstrip(".,;:! ") + "."
        story = ensure_emotion_closing(story, emotion_words or DEFAULT_EMOTION)

    return story


def extract_story_only(generated_text, prompt):
    """Remove the original prompt from the raw generated text."""
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    markers = [
        "Picture description:",
        "Requirements:",
        "Story version:",
        "Story:",
        "Observed photo caption:"
    ]
    for marker in markers:
        if marker in generated_text:
            generated_text = generated_text.split(marker)[0]

    return generated_text.strip()


def build_story_prompt(caption, base_caption, child_facts):
    """Build the prompt used by the story generation model."""
    return f"""
Write a very simple diary-style story for a young child aged 4 to 7.

Use only the photo and the child's answers.
Follow the child's answers closely.
Do not ignore the child's answers.
Do not invent unrelated people, places, events, secrets, or actions.
You may add only a small amount of simple creativity.
Use easy words.
Use short sentences.
Make it sound warm, personal, and child-friendly.
The story must clearly say: The photo showed {base_caption}.
Use the child's own ideas as the most important details.
Keep the story between 50 and 80 words.
The last sentence must describe the faces in the photo in a warm way.

Photo facts: {caption}
Child photo topic: {child_facts.get("about", "")}
Child action: {child_facts.get("doing", "")}
Child special reason: {child_facts.get("special", "")}
Child place: {child_facts.get("where", "")}
Child background: {child_facts.get("background", "")}

Story:
""".strip()


def generate_story_once(prompt, base_caption="", emotion_words="", child_facts=None, max_new_tokens=110):
    """Generate one candidate story from the language model."""
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
            temperature=0.45,
            top_p=0.78,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = extract_story_only(text, prompt)
    story = clean_story(story, base_caption=base_caption, emotion_words=emotion_words, child_facts=child_facts)
    return story


def child_facts_covered(story, child_facts):
    """Check whether the generated story includes the child's main details."""
    checks = 0
    passed = 0

    for key in ["about", "doing", "special", "where"]:
        value = child_facts.get(key, "").strip().lower()
        if value:
            checks += 1
            if value in story.lower():
                passed += 1

    if checks == 0:
        return True

    return passed >= max(1, min(2, checks))


def caption_to_story(base_caption, caption, emotion_words="", child_facts=None):
    """Generate a clean and child-friendly story, with fallback if needed."""
    child_facts = child_facts or {"background": "", "about": "", "doing": "", "special": "", "where": ""}
    if not emotion_words:
        emotion_words = DEFAULT_EMOTION

    prompt = build_story_prompt(caption, base_caption, child_facts)

    for _ in range(3):
        story = generate_story_once(
            prompt,
            base_caption=base_caption,
            emotion_words=emotion_words,
            child_facts=child_facts
        )
        story = clean_story(story, base_caption=base_caption, emotion_words=emotion_words, child_facts=child_facts)

        if not contains_unsafe_kids_content(story) and child_facts_covered(story, child_facts):
            return story

    return build_grounded_story(base_caption, emotion_words, child_facts)


# ---------- Kid scoring functions ----------

def evaluate_kid_story(kid_story, model_story):
    """Score the child's story mainly by length, with a small bonus for simple sentence structure."""
    kid_story = (kid_story or "").strip()
    model_story = (model_story or "").strip()

    if not kid_story:
        return 1

    kid_words = count_words(kid_story)
    model_words = max(1, count_words(model_story))

    if kid_words > model_words:
        score = 5
    else:
        ratio = kid_words / model_words
        if ratio >= 0.8:
            score = 4
        elif ratio >= 0.55:
            score = 3
        elif ratio >= 0.3:
            score = 2
        else:
            score = 1

    has_end_punctuation = bool(re.search(r"[.!?]\s*$", kid_story))
    has_two_sentences = len(re.findall(r"[.!?]", kid_story)) >= 2

    if score < 5 and has_end_punctuation and has_two_sentences:
        score += 1

    return max(1, min(5, score))


def build_score_message(score, animal_name):
    """Build the fun score message shown after story submission."""
    actions = ["sing for you", "dance with joy", "wave their paws", "jump up and down", "clap for you"]
    positives = ["amazing", "full of joy", "bright", "wonderful", "super lovely"]

    icon = ANIMAL_INFO[animal_name]["icon"]
    action = random.choice(actions)
    positive = random.choice(positives)

    return f"{score} {animal_name.lower()}(s) {icon} forgot to {action} because your story is too {positive}!"


# ---------- Audio functions ----------

def apply_voice_effects(audio, sr):
    """Apply a simple pitch effect to make the voice sound more playful."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()

    if audio.size == 0:
        raise ValueError("Generated audio is empty.")

    pitch_steps = 2
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)

    return np.clip(audio, -1.0, 1.0)


def text_to_audio(text):
    """Convert the final story text into audio."""
    tts = get_tts_model()
    output = tts(text)

    if "audio" not in output or "sampling_rate" not in output:
        raise ValueError("TTS model did not return valid audio output.")

    audio = np.asarray(output["audio"], dtype=np.float32).squeeze()
    sr = int(output["sampling_rate"])
    audio = apply_voice_effects(audio, sr)

    return audio, sr


def audio_to_wav_bytes(audio, sr):
    """Convert a NumPy audio array into WAV bytes for playback and download."""
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


# ---------- UI control functions ----------

def show_entry_gate():
    """Display the age warning page before the user can continue."""
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


def show_login_page():
    """Display the login page after the warning page is accepted."""
    if st.session_state.logged_in:
        return

    st.title("🐻 Little Bear Login")
    st.write("Please log in before entering the story world.")

    with st.form("login_form"):
        username = st.text_input("User name", value="", placeholder="Enter your user name")
        password = st.text_input("Password", type="password", value="", placeholder="Enter your password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.rerun()
        else:
            st.error("Wrong user name or password.")

    st.stop()


def show_sidebar_user_panel():
    """Show the logged-in user profile, total animals, and animal subtotals in the sidebar."""
    if not st.session_state.logged_in:
        return

    username = st.session_state.current_user
    total_score = get_total_score(username)
    animal_scores = get_animal_scores(username)

    st.sidebar.markdown("## 🐻 Little Bear Account")
    st.sidebar.write(f"**User name:** {username}")
    st.sidebar.write(f"**Total score:** {total_score} animals are following you!")

    st.sidebar.markdown("### Animal followers")
    for animal, meta in ANIMAL_INFO.items():
        subtotal = int(animal_scores.get(animal, 0))
        st.sidebar.write(f"{meta['icon']} {animal}: {subtotal}")

    if st.sidebar.button("Log out"):
        logout_and_return_to_warning()


def show_kid_questions():
    """Display the child input questions that guide the story generation."""
    st.markdown("### 🐻 Tell other little bears more about your photo")
    st.write("You can answer these questions to help the parent bears make a better story.")

    if st.session_state.uploaded_image_bytes is not None:
        st.caption("You can still edit these answers before asking the parent bears to tell the story.")

    st.text_area(
        "Can you share more background to other little bears about your photo?",
        key=widget_key("kid_background"),
        height=90,
        placeholder="For example: This was during our family picnic and everyone was very excited."
    )

    st.text_area(
        "What is this photo about?",
        key=widget_key("kid_photo_about"),
        height=80,
        placeholder="For example: It is about my family having fun together."
    )

    st.text_area(
        "What were you doing in the photo?",
        key=widget_key("kid_doing"),
        height=80,
        placeholder="For example: I was walking on the beach with my dad."
    )

    st.text_area(
        "Why is this photo special for you?",
        key=widget_key("kid_special"),
        height=80,
        placeholder="For example: It was the first time I told my dad my secret."
    )

    st.text_area(
        "Where was this photo taken?",
        key=widget_key("kid_where"),
        height=80,
        placeholder="For example: At HKUST, Disneyland, or at the park."
    )


def show_kid_story_scoring_area():
    """Display the child's own story box and scoring button."""
    if not st.session_state.last_story:
        return

    st.markdown("### 🐻 Write your own story")
    st.write("Now it is your turn to tell other little bears your own story.")

    kid_story = st.text_area(
        "Type your story here",
        key=widget_key("kid_own_story"),
        height=180,
        placeholder="Write your own little story here..."
    )

    if st.button("🐻 Share my stories with other little bears"):
        score = evaluate_kid_story(kid_story, st.session_state.last_story)

        if not st.session_state.score_saved_for_current_story:
            animal_name = choose_animal_for_score()
            add_score_to_user(st.session_state.current_user, score, animal_name)
            st.session_state.last_score_animal = animal_name
            st.session_state.score_saved_for_current_story = True
        else:
            animal_name = st.session_state.last_score_animal or "Bear"

        message = build_score_message(score, animal_name)
        st.session_state.kid_score = score
        st.session_state.kid_score_message = message
        st.rerun()

    if st.session_state.kid_score_message:
        st.success(st.session_state.kid_score_message)


def show_results():
    """Display the generated caption, story, audio, scoring area, and reset button."""
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

    if st.session_state.last_story:
        show_kid_story_scoring_area()

    if st.session_state.last_story or st.session_state.last_audio_bytes:
        if st.button("🐻 Make another story with a new image"):
            reset_for_another_story()


def generate_story_and_audio(image, form_values):
    """Generate the final story and voice from the uploaded image and child inputs."""
    child_facts = {
        "background": form_values["background"],
        "about": form_values["about"],
        "doing": form_values["doing"],
        "special": form_values["special"],
        "where": form_values["where"],
    }

    with st.spinner("📸 A tiny owl is peeking at your picture..."):
        base_caption, face_summary, enriched_caption, emotion_words = image_to_caption_with_expression(image, child_facts)
        if not emotion_words:
            emotion_words = DEFAULT_EMOTION

        st.session_state.last_base_caption = base_caption
        st.session_state.last_face_summary = face_summary
        st.session_state.last_caption = enriched_caption
        st.session_state.last_emotion_words = emotion_words

    with st.spinner("🍰 Your story is in the oven!"):
        story = caption_to_story(
            base_caption=base_caption,
            caption=enriched_caption,
            emotion_words=emotion_words,
            child_facts=child_facts
        )

    with st.spinner("🐻 Some little bears are tasting your story!"):
        voice_audio, sr = text_to_audio(story)
        audio_bytes = audio_to_wav_bytes(voice_audio, sr)

    st.session_state.last_story = story
    st.session_state.last_audio_bytes = audio_bytes
    st.session_state.kid_score = None
    st.session_state.kid_score_message = None
    st.session_state.score_saved_for_current_story = False
    st.session_state.last_score_animal = None


# =========================================================
# MAIN APP
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐻",
    layout="wide"
)

apply_custom_css()
ensure_default_user()
init_state()

show_entry_gate()
show_login_page()
show_sidebar_user_panel()

st.title(APP_TITLE)
st.write("Upload one image and let the parent bears turn it into a child-friendly story and voice.")

show_kid_questions()

uploaded_image = st.file_uploader(
    "Upload one image only",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    key=widget_key("uploader")
)

st.caption("Please upload only one image each time. To use a new image, click 'Make another story with a new image' first.")

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
                form_values = get_current_form_values()
                generate_story_and_audio(image, form_values)
            except RuntimeError as e:
                st.error(f"Memory or model loading issue: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    except Exception as e:
        st.error(str(e))

show_results()

if st.session_state.uploaded_image_bytes is None:
    st.info("Please upload one image to begin.")
