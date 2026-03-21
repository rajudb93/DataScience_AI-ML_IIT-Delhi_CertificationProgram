import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1. Defining the Input Text and Preparing Character Set
text = "hello world. this is a simple text generation example using rnn."
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

# 2. Creating Sequences and Labels
seq_length = 3
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

# 3. One-Hot Encoding
# Reshaping X to (number_of_sequences, seq_length, vocab_size)
X_one_hot = tf.one_hot(X, vocab_size)
y_one_hot = tf.one_hot(y, vocab_size)

# 4. Building the RNN Model
model = Sequential([
    # Input shape is (time_steps, features)
    SimpleRNN(50, input_shape=(seq_length, vocab_size), activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# 5. Compiling and Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training the model...")
model.fit(X_one_hot, y_one_hot, epochs=200, verbose=0) # Set verbose=1 to see progress

# 6. Generating New Text
start_seq = "hel" # Note: start_seq must be at least seq_length long
generated_text = start_seq

print("\nStarting generation...")
for i in range(50):
    # Take the last 'seq_length' characters to predict the next one
    current_context = [char_to_index[char] for char in generated_text[-seq_length:]]
    x_input = np.array([current_context])
    x_input_one_hot = tf.one_hot(x_input, vocab_size)
    
    # Predict and append
    prediction = model.predict(x_input_one_hot, verbose=0)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char

print("\nGenerated Text:")
print(f"'{generated_text}'")