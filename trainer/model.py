import tensorflow as tf

from trainer import dataset
from trainer import metrics
from trainer import transformer

def model_fn(features, labels, mode, params):
  """Defines how to train, evaluate and predict from the transformer model."""
  with tf.variable_scope("model"):
    model = transformer.Transformer(mode == tf.estimator.ModeKeys.TRAIN, params)
    discard_logits, riichi_logits = model(features)
    riichi_prob = tf.sigmoid(riichi_logits)

    predictions = {
        "discard_logits": discard_logits,
        "riichi_prob": riichi_prob
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # losses
    # use weight to empahsize on after riichi for defend
    is_after_riichi = tf.cast(features["is_anyone_riichi"], dtype=tf.float32)
    after_riichi_ratio = tf.reduce_mean(is_after_riichi)

    weight = 1 + is_after_riichi * params["after_riichi_instance_multiplier"]
    discard_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels["label"], logits=discard_logits)
    discard_xentropy = discard_xentropy * weight
    discard_loss = tf.reduce_mean(discard_xentropy)

    riichi_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels["is_riichi"], logits=riichi_logits)
    riichi_loss = tf.reduce_mean(riichi_xentropy)

    loss = params["riichi_loss_weight"] * riichi_loss + discard_loss

    tf.identity(discard_loss, "discard_cross_entropy")
    tf.identity(riichi_loss, "riichi_cross_entropy")
    tf.identity(loss, "total_loss")

    # evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        discard_accuracy = metrics.compute_accuracy(labels["label"], discard_logits)
        before_riichi_accuracy = discard_accuracy * (1.0 - is_after_riichi)
        after_riichi_accuracy = discard_accuracy * is_after_riichi
        eval_metrics = {
            "RIICHI_AUROC": tf.metrics.auc(labels["is_riichi"], riichi_prob, curve='ROC', summation_method='careful_interpolation'),
            "RIICHI_AUPRC": tf.metrics.auc(labels["is_riichi"], riichi_prob, curve='PR', summation_method='careful_interpolation'),
            'DISCARD_ACCURACY': tf.metrics.mean(discard_accuracy),
            'BEFORE_RIICHI_DISCARD_ACCURACY': tf.metrics.mean(before_riichi_accuracy),
            'AFTER_RIICHI_DISCARD_ACCURACY': tf.metrics.mean(after_riichi_accuracy),
            "AFTER_RIICHI_RATIO": tf.metrics.mean(after_riichi_ratio),
            "DISCARD_XENTROPY": tf.metrics.mean(discard_loss),
            "DISCARD_XENTROPY": tf.metrics.mean(discard_loss),
            "DISCARD_XENTROPY": tf.metrics.mean(discard_loss),
            "RIICHI_XENTROPY": tf.metrics.mean(riichi_loss),
            "TOTAL_LOSS": tf.metrics.mean(loss)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metrics)
    else:
        train_op, metric_dict = get_train_op_and_metrics(loss, params)
        metric_dict["minibatch_loss"] = loss
        record_scalars(metric_dict)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def serving_input_receiver_fn(params):
     def _serving_input_receiver_fn():
        input_tfrecord = tf.placeholder(dtype=tf.string, shape=[None], name="input")
        parsed_features = tf.parse_example(input_tfrecord, dataset.get_parse_spec())
        return tf.estimator.export.ServingInputReceiver(parsed_features, parsed_features)
     return _serving_input_receiver_fn


def record_scalars(metric_dict):
  for key, value in metric_dict.items():
    tf.contrib.summary.scalar(name=key, tensor=value)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    # Create a named tensor that will be logged using the logging hook.
    # The full name includes variable and names scope. In this case, the name
    # is model/get_train_op/learning_rate/learning_rate
    tf.identity(learning_rate, "learning_rate")

    return learning_rate


def get_train_op_and_metrics(loss, params):
    """Generate training op and metrics to save in TensorBoard."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            learning_rate_warmup_steps=params["learning_rate_warmup_steps"])

    # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
    # than the TF core Adam optimizer.
    optimizer = tf.contrib.opt.LazyAdamOptimizer(
        learning_rate,
        beta1=params["optimizer_adam_beta1"],
        beta2=params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    # Calculate and apply gradients using LazyAdamOptimizer.
    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    minimize_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    train_metrics = {"learning_rate": learning_rate}

    gradient_norm = tf.global_norm(list(zip(*gradients))[0])
    train_metrics["global_norm/gradient_norm"] = gradient_norm

    return train_op, train_metrics
