import tensorflow as tf

def _one_hot_columne(key, num_values):
    column = tf.feature_column.categorical_column_with_identity(key, num_values)
    column = tf.feature_column.indicator_column(column)
    return column


def _bucketized_column(key, boundaries):
    column = _numerical_column(key)
    column = tf.feature_column.bucketized_column(column, boundaries)
    column = tf.feature_column.indicator_column(column)
    return column


def _embedding_column(key, num_values, num_dim):
    column = tf.feature_column.categorical_column_with_identity(key, num_values)
    column = tf.feature_column.embedding_column(column, num_dim)
    return column


def _numerical_column(key):
    return tf.feature_column.numeric_column(key)


def get_feature_columns():
    columns = {}
    # global context
    columns["field"] = _one_hot_columne("field", 4)
    columns["round"] = _one_hot_columne("round", 4)
    columns["turn"] = _bucketized_column("turn", [24, 48])

    for name in ["center", "player0", "player1", "player2"]:
        columns["%s_oya" % name] = _numerical_column("%s_oya" % name)
        columns["%s_claim" % name] = _one_hot_columne("%s_claim" % name, 5)
        columns["%s_order" % name] = _one_hot_columne("%s_order" % name, 4)
        if "player" in name:
            columns["%s_riichi" % name] = _numerical_column("%s_riichi" % name)
    return columns


def get_context_features(parsed_features):
    all_columns = get_feature_columns()
    return tf.feature_column.input_layer(parsed_features, all_columns.values())


def get_feature_seq_bias(features, params):
    context_features = get_context_features(features)
    init_context_features = context_features
    initializer = tf.variance_scaling_initializer(
            params["initializer_gain"], mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Context_features", initializer=initializer):
        for layer_size in params['feature_hidden_size']:
              context_features = tf.layers.dense(context_features, layer_size, activation=tf.nn.relu)
        context_features = tf.layers.dense(context_features, 4, activation=None)
    context_features = tf.reshape(context_features, [-1])

    feature_seq = features["feature_seq"]
    dense_feature_seq = tf.sparse.to_dense(feature_seq)
    multiplier = tf.range(0, feature_seq.dense_shape[0]) * 4
    mask = tf.to_int64(tf.not_equal(dense_feature_seq, 0))
    feature_seq = dense_feature_seq + mask * tf.expand_dims(multiplier, axis=1)
    feature_seq_bias = tf.nn.embedding_lookup(context_features, feature_seq)
    feature_seq_bias.set_shape([None, None])
    feature_seq_bias = tf.expand_dims(tf.expand_dims(feature_seq_bias, axis=1), axis=1)
    return feature_seq_bias, init_context_features
