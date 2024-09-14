from preprare_dataset import *
from main import Transformer
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Padding
# What It Is: Padding involves adding extra tokens to sequences to make all sequences in a batch the same length. This is necessary because many models, including transformers, require inputs to be of uniform size.
# Why It’s Needed: Sequences in natural language (e.g., sentences) can have varying lengths. Padding ensures that the model can process batches of sequences efficiently, even if they are not of the same length.
# Example: If you have sentences of lengths 5, 7, and 10, you might pad them all to length 10. So, the sentences might look like [1, 2, 3, 4, 5, 0, 0, 0, 0, 0] where 0 is the padding token.

# Masking
# What It Is: Masking is a technique used to tell the model which tokens should be considered (valid) and which should be ignored (padding tokens) during processing.
# Why It’s Needed: Padding tokens are not actual content and should not influence the model’s learning or predictions. Masking helps in focusing only on the real data by ignoring these padding tokens.
# How It’s Done:
# Padding Mask: Used to ignore padding tokens in both the input sequences and the target sequences. For example, in the loss computation, padding tokens should not contribute to the loss.
# Look-ahead Mask: Used in the decoder to ensure that each token can only attend to previous tokens and not future tokens (important for autoregressive models).

# d_model
# Embedding Dimension: In the embedding layer, d_model is the size of the dense vector used to represent each token. For instance, if d_model is 512, each token in the sequence is represented by a 512-dimensional vector.
# Hidden State Dimension: During the forward pass of the Transformer, the hidden states in the encoder and decoder are also of dimension d_model. This dimension is consistent across various layers and components of the model.
# Attention Mechanism: In the multi-head self-attention mechanism, d_model influences how the queries, keys, and values are projected. Each head in multi-head attention operates in a subspace of dimension d_model / num_heads, where num_heads is the number of attention heads.
# Feed-Forward Networks: The position-wise feed-forward networks within the Transformer also operate in this space, typically using a higher-dimensional hidden layer before projecting back to d_model.
# Positional Encoding: The positional encoding vectors are also of dimension d_model, allowing the model to incorporate the positional information into the embeddings

# Embedding
# Each token in the vocabulary is usually mapped to a vector of fixed size using an embedding layer
# This vector represents the token in a continuous space, which the model uses for further processing

# Purpose of BUFFER_SIZE
# 1) Shuffling Data:
# Shuffling: Shuffling is the process of randomly rearranging the order of elements in the dataset. This helps in ensuring that the model does not learn any unintended patterns from the order of the data.
# Buffer Size: The BUFFER_SIZE specifies how many elements should be kept in memory for shuffling. For example, if BUFFER_SIZE is set to 20000, TensorFlow will randomly shuffle these 20000 elements and then sample from this shuffled set to create batches.
# 2) Impact on Performance:
# Large Buffer Size: A larger BUFFER_SIZE means more elements are shuffled at once, which generally improves randomness and the effectiveness of shuffling. However, it also requires more memory.
# Small Buffer Size: A smaller buffer may not shuffle as thoroughly, potentially leading to less effective shuffling. It uses less memory but may result in patterns in the data not being sufficiently randomized.

# Batching:
# Definition: Batching is the process of dividing your dataset into smaller, manageable chunks, or "batches." Each batch contains a fixed number of elements from the dataset.
# Purpose: This is essential for training machine learning models because it allows you to process multiple samples simultaneously. Instead of feeding the model one example at a time, you feed it a batch of examples, which makes the training process more efficient.



def create_tokenizer(vocab_size):
    # Create a TextVectorization layer with correct standardization option
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,   # Maximum vocabulary size
        standardize='lower_and_strip_punctuation',  # Corrected standardization method
        split='whitespace',       # Tokenization method
        output_mode= 'int'
    )
    return tokenizer


def tokenize(texts, tokenizer):
    # Apply tokenizer to text data
    return tokenizer(texts)


def pad_sequence(seq, max_len):
    seq_len = tf.shape(seq)[0]  # Get the current sequence length
    padding_len = max_len - seq_len  # Calculate how much padding is needed
    
    if padding_len > 0:
        # Pad with zeros (assuming 0 is the padding token)
        padding = tf.zeros([padding_len], dtype=tf.int64)  # Create the padding tensor
        seq = tf.concat([seq, padding], axis=0)  # Concatenate the sequence and the padding
    
    return seq


def get_max_sequence_length(dataset):
    max_length = 0
    
    # Iterate through the dataset and find the max length
    for batch in dataset:
        # Assuming `batch` is a tokenized sequence (could be a tuple if the dataset has inputs and targets)
        # If it's a tuple, you can access batch[0] or batch[1] depending on the structure
        sequence_length = tf.shape(batch)[1]  # Length of the sequence (assuming shape is [batch_size, seq_length])
        max_length = max(max_length, tf.reduce_max(sequence_length).numpy())
    
    return max_length


# The loss should ignore padding tokens
# We'll use a mask to compute loss only on valid tokens (non-padding tokens).
def loss_function(real, pred, mask):
    # Compute loss only on valid tokens (ignore padding)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# To calculate accuracy on the non-padded tokens
def accuracy_function(real, pred, mask):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
    mask = tf.cast(mask, dtype=accuracies.dtype)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    accuracies *= mask

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# Here, we define a single training step
# We'll use tf.GradientTape to compute gradients and apply them using the optimizer
# and calculate loss and accuracy
@tf.function
def train_step(input, target, transformer, optimizer):
    seq_len = len(input)
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        predictions = transformer.predict(input, target)
        predictions= tf.keras.layers.Input(shape= (seq_len, ), dtype= tf.int64)

        # Create a mask for non-padding tokens in the target sequence
        # Assuming 0 is the padding token
        mask = tf.cast(tf.math.not_equal(target, 0), tf.int64)

        # Calculate loss using the real target sequence and model predictions
        loss = loss_function(target, predictions, mask)

    # Calculate gradients and update the transformer model weights
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss



# During validation or testing, no gradient calculation is needed
# We'll simply evaluate the model's performance on the validation set
@tf.function
def test_step(input, target, transformer):
    # Forward pass without training
    predictions = transformer.predict(input, target)
    mask = tf.cast(tf.math.not_equal(target, 0), tf.float32)
    loss = loss_function(target, predictions, mask)
    accuracy = accuracy_function(target, predictions, mask)

    return loss, accuracy


# now we can put everything together in a loop to train the Transformer model
EPOCHS = 20
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# training set
train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# test set
val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# initilize the hyperparameters of the class Transformer 
d_model = 512
num_layers = 6
num_heads= 8
d_ff = 2048
input_vocab_size = 10000
output_vocab_size = 10000
learning_rate = 0.001

max_seq_len_train = get_max_sequence_length(train_dataset)
print(f"The maximum sequence length is: {max_seq_len_train}")

# create the optimizer, tokenizer and transformer objects 
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
tokenizer = create_tokenizer(input_vocab_size)
transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, output_vocab_size)

# Adapt the tokenizer to the dataset (train_dataset is batched, so we need to handle it carefully)
# We use map to get the input sequences (texts) from the dataset
tokenizer.adapt(train_dataset.map(lambda inp, tar: inp))


for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    total_loss = 0

    # Iterate over training batches
    # train the model with batchs from training data simultaniosly
    for batch, (inp, tar) in enumerate(train_dataset) :
        for input, target in zip(inp, tar) : # 
            print(f'Input: {input}')
            print(f'Target: {target}')

            # tokenization 
            src_token = tokenize(input, tokenizer)
            tgt_token = tokenize(target, tokenizer)

            # padding the sequences
            src_mask = pad_sequence(src_token, max_seq_len_train)
            tgt_mask = pad_sequence(tgt_token, max_seq_len_train)

            print(src_mask ,tgt_mask)
            # create the transformer object 

            batch_loss = train_step(src_mask, tgt_mask, transformer, optimizer)
            total_loss += batch_loss

            if batch % 50 == 0:
                print(f'Batch {batch} Loss {batch_loss:.4f}')
    
    # Validation after each epoch
    total_val_loss = 0
    total_val_accuracy = 0
    num_val_batches = 0

    max_seq_len_val = get_max_sequence_length(val_dataset)
    print(f"The maximum sequence length is: {max_seq_len_val}")

    for batch, (inp, tar) in enumerate(val_dataset):
        for input, target in zip(inp, tar) :

            src_token = tokenize(input, tokenizer)
            tgt_token = tokenize(target, tokenizer)
        
            src_mask = pad_sequence(src_token, max_seq_len_val)
            tgt_mask = pad_sequence(tgt_token, max_seq_len_val)
        
            val_loss, val_accuracy = test_step(src_mask, tgt_mask, transformer)
            total_val_loss += val_loss
            total_val_accuracy += val_accuracy
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    avg_val_accuracy = total_val_accuracy / num_val_batches
    print(f'Epoch {epoch + 1} Validation Loss {avg_val_loss:.4f} Accuracy {avg_val_accuracy:.4f}')

