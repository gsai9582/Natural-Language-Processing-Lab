# Lab Task 4: Morphology is an Important Factor for Word Embedding

import nltk
import fasttext
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (runs only once)
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# Sample document data
# -------------------------------
document_data = [
    "This is a sample document.",
    "Word embeddings capture morphological information.",
    "Morphology is essential for understanding words.",
    "FastText handles subword and morphology effectively."
]

# -------------------------------
# Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [
        token for token in tokens
        if token.isalpha() and token not in stop_words and token not in string.punctuation
    ]
    return " ".join(tokens)

preprocessed_data = [preprocess(sentence) for sentence in document_data]

# Save preprocessed text to file
with open("morphology_document.txt", "w", encoding="utf-8") as file:
    for line in preprocessed_data:
        file.write(line + "\n")

print("Text preprocessing completed.")
print("Preprocessed data:")
for line in preprocessed_data:
    print(line)

# -------------------------------
# Train FastText Model
# -------------------------------
model = fasttext.train_unsupervised(
    "morphology_document.txt",
    model="skipgram",
    wordNgrams=2,     # Captures morphological information
    minCount=1,
    epoch=10
)

# Save trained model
model.save_model("morphology_word_embedding.bin")

# -------------------------------
# Get Word Embedding
# -------------------------------
embedding = model.get_word_vector("morphology")

print("\nFastText Morphology Model Trained Successfully!")
print("Embedding vector length:", len(embedding))
print("First 10 values of embedding:")
print(embedding[:10])
