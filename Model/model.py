import os
import time
 
import numpy as np
import tensorflow as tf
import json
import unidecode
from keras_preprocessing.text import Tokenizer
 
print(tf.__version__)

file_path = os.path.join(os.path.dirname(__file__), 'frases.txt')
 
text = unidecode.unidecode(open(file_path, encoding='utf8').read())
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
 
encoded = tokenizer.texts_to_sequences([text])[0]
 
vocab_size = len(tokenizer.word_index) + 1
 
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

with open('word2idx_en.txt', 'w') as file:
     file.write(json.dumps(word2idx))

with open('idx2word_en.txt', 'w') as file:
     file.write(json.dumps(idx2word))

sequences = list()
 
for i in range(1, len(encoded)):
    sequence = encoded[i - 1:i + 1]
    sequences.append(sequence)

sequences = np.array(sequences)

X, Y = sequences[:, 0], sequences[:, 1]
X = np.expand_dims(X, 1)
Y = np.expand_dims(Y, 1)

# Batch size
BATCH_SIZE = 10

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 100

dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, 1]),
#    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=False),
    tf.keras.layers.Dense(rnn_units),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

embedding_dim = 10
 
units = 96
 
model = build_model(vocab_size, embedding_dim, units, BATCH_SIZE)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Directory where the checkpoints will be saved
checkpoint_dir = os.path.join(os.path.dirname(__file__), 'training_checkpoints2')

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.summary()

EPOCHS=300

#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
 
def loss_function(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
 
for epoch in range(EPOCHS):
    start = time.time()
 
    model.reset_states()
 
    for (batch, (input, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:

            predictions = model(input)

            target = tf.reshape(target, (-1,))
            loss = loss_function(target, predictions)

            grads = tape.gradient(loss, model.variables)
            model.optimizer.apply_gradients(zip(grads, model.variables))

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss.numpy().mean()))

    model.save_weights(os.path.join(checkpoint_dir, "ckpt_" + str(epoch)))

start_string = "want"
 
input_eval = [word2idx[start_string]]
input_eval = tf.expand_dims(input_eval, 0)
 
text_generated = ''

hidden = [tf.zeros((1, units))]
 
predictions = model(input_eval, hidden)
 
predictions = tf.reshape(predictions, (-1, predictions.shape[2]))

predicted_id = tf.argmax(predictions[-1]).numpy()
 
text_generated += " " + idx2word[predicted_id]
 
print(start_string + text_generated)

model.save('predictor_en.h5')