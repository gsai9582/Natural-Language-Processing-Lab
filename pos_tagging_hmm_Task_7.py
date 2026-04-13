import nltk
import pandas as pd
from nltk.corpus import treebank
from nltk.tag import hmm

# Download datasets
nltk.download('treebank')

# Training data
train_data = treebank.tagged_sents()

# Train HMM model
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train(train_data)

# Input sentence
sentence = "Natural language processing is interesting"

# Tokenization (simple & safe)
tokens = sentence.split()

# POS tagging
pos_tags = tagger.tag(tokens)

# DataFrame
df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])

print(df)
