# Import required libraries
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
from docx import Document
import string

# Download tokenizer (only first time)
nltk.download('punkt')

# Step 1: Read Word Document
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# Step 2: Preprocess Text
def preprocess_text(text):
    text = text.lower()  # convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = word_tokenize(text)  # tokenize
    return tokens

# Step 3: Generate N-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Step 4: Display N-grams with Frequency
def display_ngrams(ngram_list):
    freq = Counter(ngram_list)
    for gram, count in freq.items():
        print(f"{gram} : {count}")

# ==============================
# Main Program
# ==============================

file_path = "sample.docx"   # Replace with your Word file name

# Read document
text = read_docx(file_path)

# Preprocess text
tokens = preprocess_text(text)

print("\n===== UNIGRAM =====")
unigrams = generate_ngrams(tokens, 1)
display_ngrams(unigrams)

print("\n===== BIGRAM =====")
bigrams = generate_ngrams(tokens, 2)
display_ngrams(bigrams)

print("\n===== TRIGRAM =====")
trigrams = generate_ngrams(tokens, 3)
display_ngrams(trigrams)
