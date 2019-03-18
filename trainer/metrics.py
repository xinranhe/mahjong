import tensorflow as tf

def compute_accuracy(labels, logits):
    prediction = tf.argmax(logits, axis=1)
    max_mask = tf.one_hot(prediction, 14)
    is_acc = tf.cast(max_mask * labels > 0, dtype=tf.float32)
    return is_acc
