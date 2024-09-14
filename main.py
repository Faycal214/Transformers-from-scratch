import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np


# Attributes of the `Transformer` Class
### 1. `num_layers`
# Defines the number of layers (or "blocks") in both the encoder and decoder stacks of the Transformer. Each layer consists of multi-head attention followed by a feed-forward network.

### 2. `d_model`
# The dimensionality of the input and output embeddings, as well as the hidden states in each layer. It represents the size of the vectors that flow through the model.

### 3. `num_heads`
# Specifies how many attention heads are used in each multi-head attention block. The input and output of each layer are divided into multiple attention heads, allowing the model to focus on different parts of the input sequence.
# exemple : In a standard Transformer, `num_heads=8` indicates 8 parallel attention heads.

### 4. `d_ff`
# The dimensionality of the feed-forward network within each layer. After the multi-head attention, the output goes through a feed-forward network with hidden layers of size `d_ff`.
# exemple : If `d_ff=2048`, it indicates the feed-forward layers have 2048 units.

### 5. `input_vocab_size`
# Defines the size of the input vocabulary. It is the total number of distinct tokens or words that can appear in the input data. The input sequences will be converted into embeddings based on this vocabulary size.
# For a language model with a vocabulary of 10,000 unique words, `input_vocab_size=10000`.

### 6. `target_vocab_size`
# Similar to `input_vocab_size`, this defines the size of the target vocabulary. It is the total number of tokens or words that can appear in the output data (e.g., translations or sequences to be generated).
# For a translation task with 10,000 target words, `target_vocab_size=10000`.

### 7. `max_len`
#  Defines the maximum length of the input and output sequences. It is used to compute positional encodings, which are necessary because the model doesn't have any inherent sense of the order of tokens.
# exemple : If `max_len=5000`, the model can handle sequences of up to 5,000 tokens in length.

### **Overall Purpose of Attributes**:
# num_layers, d_model, num_heads, d_ff attributes define the complexity and capacity of the Transformer model. Larger values typically allow the model to capture more intricate patterns but increase computational cost.
# input_vocab_size, target_vocab_size, max_len attributes define the nature of the input/output data and how the model processes sequences.


class Transformer :
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_len=10000):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads    
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size= target_vocab_size
        self.max_len = max_len


    # Scaled Dot-Product Attention Layer
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute the dot products of the query (Q) and key (K) matrices.
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        # Scale the dot products by the square root of the dimensionality of the key vectors.
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply the mask to the scaled attention logits (if provided) to ignore certain positions.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Apply softmax to the scaled attention logits to get the attention weights.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # Multiply the attention weights with the value (V) matrix to get the final output.
        output = tf.matmul(attention_weights, V)
        return output, attention_weights


    # Multi-Head Attention Layer
    def multi_head_attention(self, Q, K, V, mask=None):
        # Ensure that the dimensionality of the model is divisible by the number of heads.
        assert self.d_model % self.num_heads == 0

        # Define the multi-head attention layer
        mha_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=0.1  # You can adjust dropout if needed
        )

        # Apply the multi-head attention layer
        attention_output = mha_layer(query=Q, key=K, value=V, attention_mask=mask)

        # Define a final dense layer for the output
        dense = tf.keras.layers.Dense(self.d_model)

        # Apply the final dense layer to the attention output
        output = dense(attention_output)

        return output, None


    # Feed Forward Network applied at each position
    def positionwise_feed_forward(self, x):
        # Define the first dense layer with ReLU activation.
        dense1 = tf.keras.layers.Dense(self.d_ff, activation='relu')
        # Define the second dense layer to project back to d_model dimension.
        dense2 = tf.keras.layers.Dense(self.d_model)
        # Apply the first dense layer
        x = dense1(x)
        # Apply the second dense layer
        return dense2(x)


    # Positional Encoding Layer to inject position information into the embeddings
    def positional_encoding(self):
        def get_angles(position, i, d_model):
            # Compute the angle rates for positional encoding.
            angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return position * angles

        # Calculate the angles for all positions and dimensions.
        angle_rads = get_angles(np.arange(self.max_len)[:, np.newaxis],
                                np.arange(self.d_model)[np.newaxis, :],
                                self.d_model)
        # Apply sine to even indices and cosine to odd indices.
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # Add an extra batch dimension and cast to float32.
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    

    # Encoder Layer which consists of multi-head attention and feed forward network
    def encoder_layer(self, x, dropout_rate=0.1, mask=None, training=False):
        # Apply multi-head self-attention
        mha_output, _ = self.multi_head_attention(x, x, x, mask)
        # Apply dropout and residual connection (Add & Norm block)
        mha_output = tf.keras.layers.Dropout(dropout_rate)(mha_output, training=training)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + mha_output)

        # Apply position-wise feed forward network.
        ffn_output = self.positionwise_feed_forward(out1)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output, training=training)
        # Apply another residual connection and normalization
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    

    # Decoder Layer which includes self-attention, encoder-decoder attention, and feed-forward network
    def decoder_layer(self, x, enc_output, dropout_rate=0.1, tgt_mask=None, src_mask=None, training=False):
        # Apply multi-head self-attention
        mha1_output, _ = self.multi_head_attention(x, x, x, tgt_mask)

        # apply risiduel connection and normalization to output of mha1 
        mha1_output = tf.keras.layers.Dropout(dropout_rate)(mha1_output, training=training)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + mha1_output)

        # Apply multi-head attention over the encoder output.
        mha2_output, _ = self.multi_head_attention(out1, enc_output, enc_output, src_mask)
        mha2_output = tf.keras.layers.Dropout(dropout_rate)(mha2_output, training=training)
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + mha2_output)

        # Apply position-wise feed forward network.
        ffn_output = self.positionwise_feed_forward(out2)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output, training=training)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out2 + ffn_output)
    

    # Encoder Stack: Stacks multiple Encoder layers
    def encoder(self, input, seq_len, dropout_rate=0.1):
        input = tf.keras.layers.Input(shape= (seq_len, ), dtype= tf.int64)

        # Using Lambda to calculate the dynamic shape
        # dynamic_shape_layer = tf.keras.layers.Lambda(lambda x: tf.shape(x))
        # seq_len = dynamic_shape_layer(input)

        embeddings = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)(input)

        pos_encodings = self.positional_encoding()
        pos_encodings = pos_encodings[:, :seq_len, :]

        x = embeddings + pos_encodings
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        for _ in range(self.num_layers):
            x = self.encoder_layer(x)

        return tf.keras.Model(inputs= input, outputs= x)
    
     
    # Decoder Stack: Stacks multiple Decoder layers
    def decoder(self, target, enc_output, seq_len, dropout_rate=0.1):
        target= tf.keras.layers.Input(shape= (seq_len, ), dtype= tf.int64)
        enc_output= tf.keras.layers.Input(shape= (seq_len, self.d_model), dtype= tf.float32)

        embeddings = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model)(target)

        pos_encodings = self.positional_encoding()

        pos_encodings = pos_encodings[:, :seq_len, :]
        x = embeddings + pos_encodings

        x = tf.keras.layers.Dropout(dropout_rate)(x)

        for _ in range(self.num_layers):
            x = self.decoder_layer(x, enc_output, dropout_rate)

        x = tf.keras.layers.Dense(
            self.target_vocab_size,
            activation='softmax'  # softmax activation for classification
        )(x)

        return tf.keras.Model(inputs= [target, enc_output], outputs= x)

    
    # Transformer Model: Combines encoder, decoder, and final linear projection
    def predict(self, input, target):
        input_len = len(input)
        target_len = len(target)

        input= tf.keras.layers.Input(shape= (input_len, ), dtype= tf.int64)
        target= tf.keras.layers.Input(shape= (target_len, ), dtype= tf.int64)
       
        # Get the encoder and decoder outputs
        enc_output = self.encoder(input, input_len)  # Pass input as tensor
        dec_output = self.decoder(target, enc_output, target_len)  # Pass target and encoder output as tensors
        
        enc_output= tf.keras.layers.Input(shape= (target_len, self.d_model), dtype= tf.float32)

        final_output = tf.keras.layers.Dense(self.target_vocab_size)(dec_output(inputs= [target, enc_output]))

        return tf.keras.Model(inputs=[input, target], outputs= final_output)



