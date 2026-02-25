import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
text = "Natural Language Processing is an important part of Artificial Intelligence."
print("\nOriginal Text:")
print(text)
sentences = sent_tokenize(text)
print("\nSentence Tokenization:")
print(sentences)
words = word_tokenize(text)
print("\nWord Tokenization:")
print(words)
words_lower = [w.lower() for w in words]
print("\nLowercase Words:")
print(words_lower)
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words_lower if w.isalpha() and w not in stop_words]
print("\nAfter Stopword Removal:")
print(filtered_words)
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered_words]
print("\nStemmed Words:")
print(stemmed_words)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
print("\nLemmatized Words:")
print(lemmatized_words)
pos_tags = nltk.pos_tag(filtered_words)
print("\nPOS Tagging:")
print(pos_tags)
bigrams = list(ngrams(filtered_words, 2))
print("\nBigrams:")
print(bigrams)
documents = [
    "Natural language processing is interesting",
    "Artificial intelligence includes machine learning",
    "NLP is used in many applications"
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
print("\nTF-IDF Feature Names:")
print(tfidf.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\n========== END OF NLP LAB TASK 2(A) ==========")
