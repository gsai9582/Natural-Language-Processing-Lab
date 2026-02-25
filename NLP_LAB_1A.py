# ======================================================
# NLP LAB TASK – 1(A)
# Introduction to NLP Libraries
# NLTK | spaCy | Gensim | TextBlob
# Python 3.11 Compatible (Windows)
# ======================================================

# ---------------- IMPORT NLTK ----------------
import nltk

# ---------------- OPTIONAL: spaCy ----------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    print("spaCy not installed, skipping spaCy section")
    SPACY_AVAILABLE = False

# ---------------- OPTIONAL: Gensim ----------------
try:
    from gensim import corpora
    GENSIM_AVAILABLE = True
except:
    print("gensim not installed, skipping Gensim section")
    GENSIM_AVAILABLE = False

# ---------------- OPTIONAL: TextBlob ----------------
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    print("TextBlob not installed, skipping TextBlob section")
    TEXTBLOB_AVAILABLE = False

# ---------------- NLTK DATA DOWNLOADS ----------------
nltk.download('punkt')
nltk.download('punkt_tab')                       # Python 3.11+
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # Python 3.11+

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- SAMPLE TEXT ----------------
text = "Natural Language Processing helps computers understand human language."

print("\n========== NLTK ==========")

# Sentence Tokenization
sentences = sent_tokenize(text)
print("\nSentences:")
print(sentences)

# Word Tokenization
words = word_tokenize(text)
print("\nWords:")
print(words)

# Stopword Removal (remove punctuation also)
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.isalpha() and w.lower() not in stop_words]
print("\nAfter Stopword Removal:")
print(filtered_words)

# Stemming
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered_words]
print("\nStemmed Words:")
print(stemmed_words)

# POS Tagging
pos_tags = nltk.pos_tag(filtered_words)
print("\nPOS Tags:")
print(pos_tags)

# ---------------- spaCy SECTION ----------------
if SPACY_AVAILABLE:
    print("\n========== spaCy ==========")
    doc = nlp(text)

    print("\nTokens & Lemmas:")
    for token in doc:
        print(token.text, "->", token.lemma_)

    print("\nNamed Entities:")
    for ent in doc.ents:
        print(ent.text, "-", ent.label_)

# ---------------- GENSIM SECTION ----------------
if GENSIM_AVAILABLE:
    print("\n========== GENSIM ==========")

    texts = [
        ["natural", "language", "processing"],
        ["machine", "learning"],
        ["artificial", "intelligence"]
    ]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print("\nDictionary:")
    print(dictionary.token2id)

    print("\nBag of Words:")
    print(corpus)

# ---------------- TEXTBLOB SECTION ----------------
if TEXTBLOB_AVAILABLE:
    print("\n========== TEXTBLOB ==========")

    blob = TextBlob("I love NLP. It is very useful.")

    print("\nSentiment Analysis:")
    print(blob.sentiment)

    print("\nNoun Phrases:")
    print(blob.noun_phrases)

print("\n========== END OF NLP LAB TASK 1(A) ==========")
