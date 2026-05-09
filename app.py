import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile

# =========================
# FUNCTION PART
# =========================

# ---- 1. Image → Caption (BLIP) ----
@st.cache_resource
def get_img2text_model():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

def img2text(image):
    model = get_img2text_model()
    caption = model(image)[0]["generated_text"]
    return caption


# ---- 2. Caption → Story (GPT‑2, 50–100 words, no repetition) ----
@st.cache_resource
def get_story_model():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

def text2story(caption):
    model = get_story_model()

    prompt = (
        "Write a short, imaginative story between 50 and 100 words based on the "
        f"following image description: '{caption}'. "
        "The story must be concise, non-repetitive, and written in smooth, natural English. "
        "Avoid repeating phrases or sentences. Keep it friendly and suitable for all ages.\n\nStory:"
    )

    output = model(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.8
    )[0]["generated_text"]

    # Remove the prompt from the output
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    # Clean up repeated or incomplete sentences
    sentences = output.split(".")
    cleaned = ". ".join(
        s.strip() for s in sentences
        if len(s.strip()) > 0
    )
    cleaned = cleaned.strip() + "."

    return cleaned


# ---- 3. Story → Audio (gTTS) ----
def text2audio(story_text):
    tts = gTTS(story_text)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name  # return path to audio file


# =========================
# MAIN PART (Streamlit UI)
# =========================

st.title("📖 Image Storytelling App")
st.write("Upload an image → get a caption → generate a story → listen to the audio narration.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
    except Exception:
        st.error("Invalid image file. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Story"):
        # Step 1: Caption
        with st.spinner("Generating caption..."):
            caption = img2text(image)
        st.success("Caption generated!")
        st.write(f"**Caption:** {caption}")

        # Step 2: Story
        with st.spinner("Writing story..."):
            story = text2story(caption)
        st.success("Story created!")
        st.write("### 📘 Your Story")
        st.write(story)

        # Step 3: Audio
        with st.spinner("Generating audio..."):
            audio_path = text2audio(story)

        st.success("Audio ready!")
        st.audio(audio_path)

    if st.button("Clear"):
        st.experimental_rerun()



