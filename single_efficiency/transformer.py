# copied and adapted from https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
import tensorflow as tf

from single_efficiency import attention_layer
from single_efficiency import embedding_layer
from single_efficiency import ffn_layer


class Transformer(object):
    """
    Transformer model for japanese mahjong discarded hai prediction.
    """
    def __init__(self, train, params):
        self.train = train
        self.params = params
        self.encoder_stack = EncoderStack(params, train)
        self.hai_emb_layer = embedding_layer.EmbeddingSharedWeights("hai_emb", 34, 16)

    def __call__(self, input):
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        
        with tf.variable_scope("Transformer", initializer=initializer):
            encoder_inputs = self.hai_emb_layer(input)
            if self.train:
                encoder_inputs = tf.nn.dropout(encoder_inputs, 1 - self.params["layer_postprocess_dropout"])

            inputs_padding = tf.zeros_like(input, dtype=tf.float32)
            attention_bias = tf.expand_dims(tf.expand_dims(inputs_padding, axis=1), axis=1)

            encoder_outputs = self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

            value = tf.reduce_sum(tf.layers.dense(encoder_outputs, 1), axis=1)
        return value


class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params, train):
    self.layer = layer
    self.postprocess_dropout = params["layer_postprocess_dropout"]
    self.train = train

    # Create normalization layer
    self.layer_norm = LayerNormalization(params["hidden_size"])

  def __call__(self, x, *args, **kwargs):
    # Preprocessing: apply layer normalization
    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if self.train:
      y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
    return x + y


class EncoderStack(tf.layers.Layer):
  """Transformer encoder stack.
  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params, train):
    super(EncoderStack, self).__init__()
    self.layers = []
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"], train)
      feed_forward_network = ffn_layer.FeedFowardNetwork(
          params["hidden_size"], params["filter_size"],
          params["relu_dropout"], train, params["allow_ffn_pad"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params, train),
          PrePostProcessingWrapper(feed_forward_network, params, train)])

    # Create final layer normalization layer.
    self.output_normalization = LayerNormalization(params["hidden_size"])

  def call(self, encoder_inputs, attention_bias, inputs_padding):
    """Return the output of the encoder layer stacks.
    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer.
        [batch_size, 1, 1, input_length]
      inputs_padding: P
    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.variable_scope("layer_%d" % n):
        with tf.variable_scope("self_attention"):
          encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
          encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

    return self.output_normalization(encoder_inputs)
