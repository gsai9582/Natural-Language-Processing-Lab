# ======================================================
# NLP LAB TASK – 2(B)
# Tokenization using Transformers Package
# ======================================================

# ---------- INSTALL (Run once if needed) ----------
# pip install transformers nltk

# ---------- IMPORT LIBRARIES ----------
from transformers import AutoTokenizer, BertTokenizer
import nltk
from nltk.corpus import stopwords

# ---------- DOWNLOAD NLTK DATA ----------
nltk.download('stopwords')

# ---------- LOAD TOKENIZER ----------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\n========== TASK 2(B): TRANSFORMER TOKENIZATION ==========")

# ======================================================
# (1) BASIC TOKENIZATION
# ======================================================
text1 = "Hello, how are you doing today?"
tokens = tokenizer.tokenize(text1)

print("\n1. Basic Tokenization:")
print(tokens)

# ======================================================
# (2) ENCODING – TOKEN IDS
# ======================================================
token_ids = tokenizer.encode(text1)
print("\n2. Token IDs (Encoding):")
print(token_ids)

# ======================================================
# (3) TOKENIZATION WITH MAX LENGTH & TRUNCATION
# ======================================================
tokens_limited = tokenizer(text1, max_length=64, truncation=True)

print("\n3. Tokenization with Max Length & Truncation:")
print(tokens_limited)

# ======================================================
# (4) TOKENIZATION WITH STOPWORDS REMOVED
# ======================================================
stop_words = set(stopwords.words("english"))
text2 = "This is a sample sentence with some stopwords."

filtered_tokens = [
    token for token in tokenizer.tokenize(text2)
    if token.lower() not in stop_words
]

print("\n4. Tokenization after Stopword Removal:")
print(filtered_tokens)

# ======================================================
# (5) STOPWORDS AS DELIMITERS
# ======================================================
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

filtered_text = ' '.join(
    [word if word.lower() not in stop_words else '[STOP]' for word in text2.split()]
)

tokens_with_delimiters = bert_tokenizer.tokenize(filtered_text)

print("\n5. Stopwords Replaced with Delimiters:")
print(tokens_with_delimiters)

# ======================================================
# (6) CUSTOM STOPWORDS TOKENIZATION
# ======================================================
custom_stopwords = ["hello", "are", "you", "doing", "today"]
text3 = "Hello, how are you doing today?"

custom_tokens = []
for word in text3.split():
    if word.lower() not in custom_stopwords:
        custom_tokens.extend(tokenizer.tokenize(word))

print("\n6. Tokenization using Custom Stopwords:")
print(custom_tokens)

print("\n========== END OF NLP LAB TASK 2(B) ==========")
