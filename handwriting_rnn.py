"""
Handwritten Text Generation (Character-level RNN)
----------------------------------------------------
Trains a character-level LSTM (a type of RNN) on a text dataset and
uses it to generate brand-new text in a similar style, character by
character.

Dataset expected: input.txt (Tiny Shakespeare, or any plain text file)

Requirements:
    pip install tensorflow numpy

Usage:
    python handwriting_rnn.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input

DATA_FILE = "input.txt"
SEQ_LENGTH = 100       # how many characters the model looks at to predict the next one
BATCH_SIZE = 64
EPOCHS = 20            # increase this for better results (takes longer)
EMBEDDING_DIM = 64
RNN_UNITS = 256


def main():
    # -------------------------------------------------------------
    # 1. Load and inspect the text
    # -------------------------------------------------------------
    print("Loading text...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  -> Total characters: {len(text)}")

    # Unique characters in the text -- this becomes our "vocabulary"
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    print(f"  -> Unique characters (vocab size): {vocab_size}")

    # Map each character to a number and back
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    # Convert the entire text into a sequence of numbers
    text_as_int = np.array([char_to_idx[ch] for ch in text])

    # -------------------------------------------------------------
    # 2. Build training sequences
    # -------------------------------------------------------------
    # Each training example: SEQ_LENGTH characters as input,
    # the next SEQ_LENGTH characters (shifted by 1) as target.
    print("\nBuilding training sequences...")
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)

    # -------------------------------------------------------------
    # 3. Build the model
    # -------------------------------------------------------------
    # Embedding -> turns each character index into a learnable vector
    # LSTM       -> the actual recurrent layer that learns sequence patterns
    # Dense      -> outputs a probability for every character in the vocab
    print("\nBuilding model...")
    model = Sequential([
        Input(batch_shape=(BATCH_SIZE, None)),
        Embedding(vocab_size, EMBEDDING_DIM),
        LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        Dense(vocab_size),
    ])
    model.summary()

    def loss_fn(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer="adam", loss=loss_fn)

    # -------------------------------------------------------------
    # 4. Train
    # -------------------------------------------------------------
    print("\nTraining... (this can take a while, especially without a GPU)")
    model.fit(dataset, epochs=EPOCHS)

    # -------------------------------------------------------------
    # 6. Rebuild model with batch_size=1 for generation
    # -------------------------------------------------------------
    # (The trained model expects batches of 64; for generating text we
    # only need to process ONE sequence at a time, so we copy the
    # learned weights into a batch_size=1 version of the same model.)
    gen_model = Sequential([
        Input(batch_shape=(1, None)),
        Embedding(vocab_size, EMBEDDING_DIM),
        LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        Dense(vocab_size),
    ])
    gen_model.set_weights(model.get_weights())

    # -------------------------------------------------------------
    # 7. Generate final text samples
    # -------------------------------------------------------------
    print("\n=========== FINAL GENERATED TEXT ===========")
    print(generate_text(gen_model, "ROMEO: ", char_to_idx, idx_to_char, vocab_size, length=500))
    print("==============================================")

    gen_model.save_weights("handwriting_rnn_weights.weights.h5")
    print("\nModel weights saved -> handwriting_rnn_weights.weights.h5")


def generate_text(model, start_string, char_to_idx, idx_to_char, vocab_size, length=300, temperature=1.0):
    """
    Generates new text character by character.
    temperature: lower = safer/more repetitive, higher = more random/creative
    """
    input_eval = [char_to_idx[ch] for ch in start_string if ch in char_to_idx]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    for layer in model.layers:
        if hasattr(layer, "reset_state"):
            layer.reset_state()

    for _ in range(length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])

    return start_string + "".join(text_generated)


if __name__ == "__main__":
    main()