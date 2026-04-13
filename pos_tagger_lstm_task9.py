# ==========================================
# Task 9: POS Tagging (Simple Version)
# Works WITHOUT TensorFlow
# ==========================================

import re

# Input (unstructured text)
raw_text = "<html>This is a book. He reads a book daily.</html>"

# Remove HTML
clean_text = re.sub(r'<.*?>', '', raw_text)

# Lowercase + tokenize
words = clean_text.lower().replace('.', '').split()

# Simple POS rules (simulation)
def simple_pos(word):
    if word in ['this', 'a']:
        return 'DT'
    elif word in ['he']:
        return 'PRP'
    elif word.endswith('s'):
        return 'VB'
    elif word in ['book']:
        return 'NN'
    elif word.endswith('ly'):
        return 'RB'
    else:
        return 'NN'

# Tagging
tagged = [(w, simple_pos(w)) for w in words]

print("POS Tagged Output:")
print(tagged)

# Extract nouns
info = [w for w, t in tagged if t == 'NN']

print("\nExtracted Information (Nouns):")
print(info)
