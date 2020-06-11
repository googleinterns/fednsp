"""This file implements a transformer-based model to detect intents and the
corresponding slots for a given query.

The model consists of the transformer encoder layer that computes the
contextual embedding of the input tokens. The intent is predicted by
passing a concatenation of all the contextual embeddings though a
linear layer. The decoder is used to predict the slot tokens. Most of
the code was in this file was taken from
https://www.tensorflow.org/tutorials/text/transformer.

  Typical usage example:

    model = Net(num_layers=3, d_model=128, num_heads=8, dff=512, \
      input_vocab_size=800, intent_vocab_size=50, slot_vocab_size=100, \
      pe_max=64, max_seq_len=48)
"""

import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
    """Computes the angles for all the input positions which will then be used
    to compute the positional encodings.

    Args:
    pos: a numpy array of the position indices
    i: a numpy array of the embedding indexes
    d_model: the dimensionality of the embeddings

    Returns:
    angles
    """

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """Computes the positional encodings for all the input positions. These
    embeddings are a function of the embedding dimension.

    Args:
    position: the maximum index upto which the positional encoding is to be computed
    d_model: the dimensionality of the embeddings

    Returns:
    pos_encoding
    """

    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(query, key, value, mask):
    """Calculates the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    query: query shape == (..., seq_len_q, depth)
    key: key shape == (..., seq_len_k, depth)
    value: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(query, key,
                          transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dkey = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dkey)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Defines the Multi-Headed Attention layer.

    This class defines the tensorflow layers and helper function required
    to perform multi-headed attention.

    Attributes:
        num_heads: The number of heads on which attention is computed on.
        d_model: The dimensionality of the contextual embeddings.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads
        self.wquery = tf.keras.layers.Dense(self.d_model)
        self.wkey = tf.keras.layers.Dense(self.d_model)
        self.wvalue = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, combined_input, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        split_output = tf.reshape(combined_input,
                                  (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(split_output, perm=[0, 2, 1, 3])

    def call(self, value, key, query, mask):
        """Forward pass for the Multi-Head Attention layer.
        """

        batch_size = tf.shape(query)[0]

        query = self.wquery(query)  # (batch_size, seq_len, d_model)
        key = self.wkey(key)  # (batch_size, seq_len, d_model)
        value = self.wvalue(value)  # (batch_size, seq_len, d_model)

        query = self.split_heads(
            query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(
            key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(
            value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value, mask)

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(
            concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """Defines the feed forward component of the transformer.

    The feed forward layer consists of a dense two dense layers.

    Args:
    d_model: The dimensionality of the embeddings.
    dff: The output dimension of the first dense layer.

    Returns:
    keras layers
    """

    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff,
                              activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """Defines the Encoder layer of the transformer.

    This class defines the tensorflow layers and the forward pass for the
    transformer decoder layer.

    Attributes:
        d_model: The dimensionality of the contextual embeddings.
        num_heads: The number of heads on which attention is computed on.
        dff: The hidden dimension of the feed forward component of the
            transformer.
        rate: The dropout rate to be used.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, mask):
        """Forward pass for the Encoder layer.
        """

        attn_output, _ = self.mha(inputs, inputs, inputs,
                                  mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """Defines the Decoder layer of the transformer.

    This class defines the tensorflow layers and the forward pass for the
    transformer decoder layer.

    Attributes:
        d_model: The dimensionality of the contextual embeddings.
        num_heads: The number of heads on which attention is computed on.
        dff: The hidden dimension of the feed forward component of the
            transformer.
        rate: The dropout rate to be used.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, enc_output, training, look_ahead_mask,
             padding_mask):
        """Forward pass for the Decoder layer.
        """

        attn1, attn_weights_block1 = self.mha1(
            inputs, inputs, inputs,
            look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1,
            padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 +
                               out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output +
                               out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """Defines the Encoder component of the transformer.

    The encoder of a transformer consists of multiple encoder layers
    stacked on top of each other. This class defines the tensorflow
    layers and the forward pass for the encoder.

    Attributes:
        num_layers: The number of encoder layers in the encoder.
        d_model: The dimensionality of the contextual embeddings.
        num_heads: The number of heads on which attention is computed on.
        dff: The hidden dimension of the feed forward component of the
            transformer.
        input_vocab_size: The size of the input vocabulary.
        maximum_positional_encoding: The maximum number of input position.
        rate: The dropout rate to be used.
    """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, mask):
        """Forward pass for the Encoder.
        """
        seq_len = tf.shape(inputs)[1]

        # adding embedding and position encoding.
        inputs = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :seq_len, :]

        output = self.dropout(inputs, training=training)

        for i in range(self.num_layers):
            output = self.enc_layers[i](output, training, mask)

        return output  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """Defines the Decoder component of the transformer.

    The decoder of a transformer consists of multiple decoder layers
    stacked on top of each other. This class defines the tensorflow
    layers and the forward pass for the decoder.

    Attributes:
        num_layers: The number of encoder layers in the decoder.
        d_model: The dimensionality of the contextual embeddings.
        num_heads: The number of heads on which attention is computed on.
        dff: The hidden dimension of the feed forward component of the
            transformer.
        input_vocab_size: The size of the input vocabulary.
        maximum_positional_encoding: The maximum number of input position.
        rate: The dropout rate to be used.
    """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, enc_output, training, look_ahead_mask,
             padding_mask):
        """Forward pass for the Decoder.
        """

        seq_len = tf.shape(inputs)[1]
        attention_weights = {}

        inputs = self.embedding(
            inputs)  # (batch_size, target_seq_len, d_model)
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :seq_len, :]

        output = self.dropout(inputs, training=training)

        for i in range(self.num_layers):
            output, block1, block2 = self.dec_layers[i](output, enc_output,
                                                        training,
                                                        look_ahead_mask,
                                                        padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return output, attention_weights


class SlotHead(tf.keras.layers.Layer):
    """Defines the SlotHead which computes the slot for each input token.

    The slot head uses a linear layer on top of the contextual embeddings
    from the transformer encoder to predict the slots for each input token.

    Attributes:
        slot_vocab_size: The size of the slot vocabulary.
    """
    def __init__(self, slot_vocab_size):
        super(SlotHead, self).__init__()

        self.slot_layer = tf.keras.layers.Dense(slot_vocab_size)

    def call(self, inputs):
        """Forward pass for the Slot Head.
        """

        out = self.slot_layer(inputs)

        return out  # (batch_size, input_seq_len, slot_vocab_size)


class IntentHead(tf.keras.layers.Layer):
    """Defines the IntentHead which computes the intent of the given input.

    The intent head uses a linear layer on the concatenation of all the
    contextual embeddings for the given input to predict the intent.

    Attributes:
        intent_vocab_size: The size of the intent vocabulary.
        d_model: The dimensionality of the embeddings.
        seq_len: The sequence length of the inputs to the transformer.
    """
    def __init__(self, intent_vocab_size, d_model, seq_len):
        super(IntentHead, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.intent_layer = tf.keras.layers.Dense(intent_vocab_size)

    def call(self, inputs, mask):
        """Forward pass for the Intent head.
        """

        batch_size = tf.shape(inputs)[0]

        #Apply masking to remove padding embeddings
        inputs = tf.multiply(inputs, mask)

        inputs = tf.reshape(
            inputs, (batch_size, self.seq_len *
                     self.d_model))  # (batch_size, input_seq_len*d_model)
        out = self.intent_layer(inputs)

        return out  # (batch_size, intent_vocab_size)


class Net(tf.keras.Model):
    """Defines the transformer based network to predict slots and intents.

    The network consists of an encoder followed by a decoder. A linear layer
    is used on top of the encoder output to predict the intent. Another linear
    layer is used on top of the decoder output to predict the slots labels.

    Attributes:
        num_layers: The number of transformer encoder layers.
        d_model: The dimensionality of the contextual embeddings.
        num_heads: The number of heads on which attention is computed on.
        dff: The hidden dimension of the feed forward component of the
            transformer.
        input_vocab_size: The size of the input vocabulary.
        slot_vocab_size: The size of the slot vocabulary.
        intent_vocab_size: The size of the intent vocabulary.
        pe_max: The maximum number of positional embeddings needed.
        max_seq_len: The maimum sequence length of the input.
        rate: The dropout rate to be used.
    """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 slot_vocab_size,
                 intent_vocab_size,
                 pe_max,
                 max_seq_len=48,
                 rate=0.1):
        super(Net, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_max, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               slot_vocab_size, pe_max, rate)

        self.intent_head = IntentHead(intent_vocab_size, d_model, max_seq_len)

        self.slot_head = SlotHead(slot_vocab_size)

    def call(self,
             inputs,
             targets,
             training,
             padding_mask,
             look_ahead_mask,
             intent_mask=None):
        """Forward pass for the entire model.
        """

        enc_output = self.encoder(
            inputs, training,
            padding_mask)  # (batch_size, inp_seq_len, d_model)

        dec_output, _ = self.decoder(targets, enc_output,
                                     training, look_ahead_mask,
                                     padding_mask)

        intent_output = self.intent_head(
            enc_output,
            intent_mask)  # (batch_size, tar_seq_len, intent_vocab_size

        slot_output = self.slot_head(
            dec_output)  # (batch_size, tar_seq_len, slot_vocab_size)

        return slot_output, intent_output
