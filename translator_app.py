import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import uuid

st.set_page_config(page_title="AI Translator", layout="centered")

st.title("🌐 English to Hindi Translator")
st.markdown("Enter your English text below and get a Hindi translation with optional audio.")

@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()


def translate_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    output = model.generate(**tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

text_input = st.text_area("Enter English Text:")

if st.button("Translate"):
    if text_input.strip():
        with st.spinner("Translating..."):
            translated = translate_text(text_input)
        st.success("Translated Text (Hindi):")
        st.markdown(f"**{translated}**")

        if st.button("🔊 Listen"):
            tts = gTTS(text=translated, lang="hi")
            filename = f"{uuid.uuid4()}.mp3"
            tts.save(filename)
            st.audio(filename, format="audio/mp3")
    else:
        st.warning("Please enter some text.")
