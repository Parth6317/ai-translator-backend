from transformers import MarianMTModel, MarianTokenizer

# ✅ This model works for English to Hindi
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text: str) -> str:
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
