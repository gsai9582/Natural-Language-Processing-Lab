# ==========================================
# Task 10: Chunking (Rule-Based Version)
# ==========================================

import re

# Step 1: Input (unstructured text)
raw_text = "<html>The quick brown fox jumps over the lazy dog</html>"

# Step 2: Preprocessing
clean_text = re.sub(r'<.*?>', '', raw_text)
words = clean_text.lower().split()

# Step 3: Simple POS tagging (rule-based)
def pos_tag(word):
    if word in ['the']:
        return 'DT'
    elif word in ['quick', 'brown', 'lazy']:
        return 'JJ'
    elif word in ['fox', 'dog']:
        return 'NN'
    elif word in ['jumps']:
        return 'VB'
    else:
        return 'NN'

pos_tags = [(w, pos_tag(w)) for w in words]

# Step 4: Chunking (NP: DT + JJ + NN)
chunks = []
current_chunk = []

for word, tag in pos_tags:
    if tag in ['DT', 'JJ', 'NN']:
        current_chunk.append((word, tag))
    else:
        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
        chunks.append([(word, tag)])

if current_chunk:
    chunks.append(current_chunk)

# Step 5: Output
print("POS Tagged Text:")
print(pos_tags)

print("\nChunks:")
for chunk in chunks:
    print(chunk)

# Step 6: Extract Noun Phrases
noun_phrases = [" ".join(w for w, t in chunk) 
                for chunk in chunks if chunk[0][1] == 'DT']

print("\nNoun Phrases:")
print(noun_phrases)
