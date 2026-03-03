import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
from docx import Document
import string

nltk.download('punkt')

# ----------------------------
# Step 1: Read Word Document
# ----------------------------
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# ----------------------------
# Step 2: Preprocess Text
# ----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens

# ----------------------------
# Step 3: Generate N-grams
# ----------------------------
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# ----------------------------
# Step 4: Laplace (Add-1) Smoothing
# ----------------------------
def laplace_smoothing(ngram_counts, prefix_counts, vocab_size):
    smoothed_probs = {}
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        prefix_count = prefix_counts[prefix]
        smoothed_probs[ngram] = (count + 1) / (prefix_count + vocab_size)
    return smoothed_probs

# ----------------------------
# Step 5: Add-k Smoothing
# ----------------------------
def add_k_smoothing(ngram_counts, prefix_counts, vocab_size, k):
    smoothed_probs = {}
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        prefix_count = prefix_counts[prefix]
        smoothed_probs[ngram] = (count + k) / (prefix_count + k * vocab_size)
    return smoothed_probs

# =============================
# MAIN PROGRAM
# =============================
file_path = "sample.docx"   # Keep Word file in same folder

# Read and preprocess
text = read_docx(file_path)
tokens = preprocess(text)

# Generate BIGRAMS
bigrams = generate_ngrams(tokens, 2)

# Count bigrams and unigrams (prefix)
bigram_counts = Counter(bigrams)
unigram_counts = Counter(tokens)

vocab_size = len(set(tokens))

print("\n========== ORIGINAL BIGRAM PROBABILITIES ==========")
for bigram, count in bigram_counts.items():
    prefix = bigram[0]
    original_prob = count / unigram_counts[prefix]
    print(f"{bigram} -> {original_prob:.4f}")

# Laplace Smoothing
laplace_probs = laplace_smoothing(bigram_counts, unigram_counts, vocab_size)

print("\n========== LAPLACE (ADD-1) SMOOTHING ==========")
for bigram, prob in laplace_probs.items():
    print(f"{bigram} -> {prob:.4f}")

# Add-k Smoothing (k=0.5 example)
k = 0.5
addk_probs = add_k_smoothing(bigram_counts, unigram_counts, vocab_size, k)

print("\n========== ADD-k SMOOTHING (k=0.5) ==========")
for bigram, prob in addk_probs.items():
    print(f"{bigram} -> {prob:.4f}")
