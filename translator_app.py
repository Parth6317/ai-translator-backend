import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import uuid

st.set_page_config(page_title="AI Translator", layout="centered")

st.title("🌐 English Translator")
st.markdown("Enter English text, select a language, and get the translation .")

# Language model mappings
language_options = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Gujarati": "Helsinki-NLP/opus-mt-en-guj"
}


# Select language
target_lang = st.selectbox("Choose target language:", list(language_options.keys()))

# Load model/tokenizer based on selected language
model_name = language_options[target_lang]
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    output = model.generate(**tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Input box
text_input = st.text_area("Enter English Text:")

# Translation
if st.button("Translate"):
    if text_input.strip():
        with st.spinner("Translating..."):
            translated = translate_text(text_input)
        st.success(f"Translated Text ({target_lang}):")
        st.markdown(f"**{translated}**")

        # Text-to-Speech
        if st.button("🔊 Listen"):
            tts_lang = "hi" if target_lang == "Hindi" else "gu"
            tts = gTTS(text=translated, lang=tts_lang)
            filename = f"{uuid.uuid4()}.mp3"
            tts.save(filename)
            st.audio(filename, format="audio/mp3")
    else:
        st.warning("Please enter some text.")
