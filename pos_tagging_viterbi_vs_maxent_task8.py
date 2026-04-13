# ==========================================
# Task 8: POS Tagging using Viterbi & Log-linear Model
# FINAL CORRECT VERSION
# ==========================================

import nltk
import random
import warnings

# Ignore warnings (clean output)
warnings.filterwarnings("ignore")

# Download dataset (run once)
nltk.download('treebank')

from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.tag import ClassifierBasedPOSTagger
from nltk.probability import LidstoneProbDist

# ------------------------------------------
# Load Dataset
# ------------------------------------------
data = list(treebank.tagged_sents())
random.shuffle(data)

# Use smaller dataset for fast execution
train_data = data[:2000]
test_data = data[2000:2200]

# ------------------------------------------
# 1. HMM MODEL (VITERBI with smoothing)
# ------------------------------------------
print("\n--- HMM Model (Viterbi) ---")

trainer = hmm.HiddenMarkovModelTrainer()

# Apply smoothing (IMPORTANT FIX)
hmm_tagger = trainer.train(
    train_data,
    estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)
)

# Accuracy
hmm_accuracy = hmm_tagger.accuracy(test_data)
print("HMM Accuracy:", round(hmm_accuracy * 100, 2), "%")

# Test sentence (safe tokenization)
sentence = "The dog barks loudly".split()
print("HMM Tagging:", hmm_tagger.tag(sentence))


# ------------------------------------------
# 2. LOG-LINEAR MODEL (MAXENT)
# ------------------------------------------
print("\n--- Log-linear Model (MaxEnt) ---")

# Feature extraction
def features(sentence, index):
    word = sentence[index]
    return {
        'word': word,
        'lower': word.lower(),
        'suffix-3': word[-3:],
        'prefix-2': word[:2],
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'prev_word': '' if index == 0 else sentence[index-1]
    }

# Train MaxEnt model (reduced data for speed)
maxent_tagger = ClassifierBasedPOSTagger(
    train=train_data[:1000],
    feature_detector=features
)

# Accuracy
maxent_accuracy = maxent_tagger.accuracy(test_data)
print("MaxEnt Accuracy:", round(maxent_accuracy * 100, 2), "%")

# Test sentence
print("MaxEnt Tagging:", maxent_tagger.tag(sentence))


# ------------------------------------------
# 3. COMPARISON
# ------------------------------------------
print("\n--- Comparison ---")

print("HMM Accuracy     :", round(hmm_accuracy * 100, 2), "%")
print("MaxEnt Accuracy  :", round(maxent_accuracy * 100, 2), "%")

if maxent_accuracy > hmm_accuracy:
    print("Result: Log-linear model performs better.")
else:
    print("Result: HMM performs better.")
