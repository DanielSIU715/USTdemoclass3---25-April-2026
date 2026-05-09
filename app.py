import streamlit as st
from transformers import pipeline
from PIL import Image

# =========================
# Function part
# =========================

@st.cache_resource
def get_img2text_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

@st.cache_resource
def get_story_model():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

@st.cache_resource
def get_tts_model():
    return pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng"
    )


def img2text(image):
    img2txt = get_img2text_model()
    caption = img2txt(image)[0]["generated_text"]
    return caption


def text2story(caption):
    story_model = get_story_model()

    prompt = (
        "You are a friendly storyteller. "
        "Write a short, imaginative, heartwarming story based on this description: "
        f"'{caption}'. "
        "The story should be easy to understand, suitable for all ages, and around 3–5 paragraphs."
    )

    out = story_model(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )[0]["generated_text"]

    # Optionally trim the prompt from the beginning
    if out.startswith(prompt):
        out = out[len(prompt):].strip()

    return out


def text2audio(story_text):
    tts = get_tts_model()
    audio_out = tts(story_text)
    audio_array = audio_out["audio"]
    sample_rate = audio_out["sampling_rate"]
    return audio_array, sample_rate


# =========================
# Main part (Streamlit UI)
# =========================

st.title("📖 Image Storytelling App")
st.write("Upload an image, get a story, and listen to it as audio.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
    except Exception:
        st.error("Invalid image file. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Story"):
        # 1. Image → Caption
        with st.spinner("Analyzing image and generating caption..."):
            caption = img2text(image)
        st.success("Caption generated!")
        st.write(f"**Caption:** {caption}")

        # 2. Caption → Story
        with st.spinner("Writing story..."):
            story = text2story(caption)
        st.success("Story created!")
        st.write("### 📘 Your Story")
        st.write(story)

        # 3. Story → Audio
        with st.spinner("Generating audio..."):
            audio_array, sample_rate = text2audio(story)

        st.success("Audio ready!")
        st.audio(audio_array, sample_rate=sample_rate)

    if st.button("Clear"):
        st.experimental_rerun()


