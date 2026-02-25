# ======================================================
# NLP LAB – TASK 3
# Single Layer LSTM for Word Generation
# ======================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------
# 1. Input Text Data
# ------------------------------------------------------
text = """
Natural language processing enables machines to understand human language.
LSTM models are powerful for sequence learning and text generation.
"""

# ------------------------------------------------------
# 2. Tokenization
# ------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
token_list = tokenizer.texts_to_sequences([text])[0]

for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

# Padding
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# ------------------------------------------------------
# 3. Build Single Layer LSTM Model
# ------------------------------------------------------
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
model.add(LSTM(100))              # ✅ Single LSTM Layer
model.add(Dense(total_words, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------
# 4. Train the Model
# ------------------------------------------------------
model.fit(X, y, epochs=200, verbose=1)

# ------------------------------------------------------
# 5. Text Generation
# ------------------------------------------------------
seed_text = "natural language"
next_words = 5

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

    seed_text += " " + output_word

print("\nGenerated Text:")
print(seed_text)
