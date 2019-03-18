import tensorflow as tf

def _embedding_column(key, num_values, num_dim):
    column = tf.feature_column.categorical_column_with_identity(key, num_values)
    column = tf.feature_column.embedding_column(column, num_dim)
    return column


def _numerical_column(key):
    return tf.feature_column.numeric_column(key)


def get_feature_columns():
    columns = {}
    # numerical columns
    columns["center_oya"] = _numerical_column("center_oya")
    for pid in xrange(3):
        columns["player%d_oya" % pid] = _numerical_column("player%d_oya" % pid)
        columns["player%d_riichi" % pid] = _numerical_column("player%d_riichi" % pid)
    
    # individual embedding columes
    columns["round"] = _embedding_column("round", 4, 2)
    for pid in xrange(3):
        columns["player%d_claim" % pid] = _embedding_column("player%d_claim" % pid, 5, 2)
        columns["player%d_order" % pid] = _embedding_column("player%d_order" % pid, 7, 4)
        columns["player%d_score" % pid] = _embedding_column("player%d_score" % pid, 161, 15)
        
    # shared_embedding column
    input_columns = []
    input_columns.append(tf.feature_column.categorical_column_with_identity("current_field", 4))
    input_columns.append(tf.feature_column.categorical_column_with_identity("center_field", 4))
    for pid in xrange(3):
        input_columns.append(tf.feature_column.categorical_column_with_identity("player%d_field" % pid, 4))
    shared_embedding_columns = tf.feature_column.shared_embedding_columns(input_columns, 2)
    columns["current_field"] = shared_embedding_columns[0]
    columns["center_field"] = shared_embedding_columns[1]
    for pid in xrange(3):
        columns["player%d_field" % pid] = shared_embedding_columns[pid + 2]
    return columns


def get_context_embedding_mtx(parsed_features):
    """
    Takes input features and returns the player context features
    embedding lookup matrix.
    
    returns: context_embedding_mtx of size (batch, 4, embedding_dim (32)) 
    """
    all_columns = get_feature_columns()
    features = []
    for pid in xrange(3):
        columns = [
            all_columns["center_oya"],
            all_columns["round"],
            all_columns["current_field"],
            all_columns["center_field"],
            all_columns["player%d_oya" % pid],
            all_columns["player%d_riichi" % pid],
            all_columns["player%d_claim" % pid],
            all_columns["player%d_order" % pid],
            all_columns["player%d_score" % pid],
            all_columns["player%d_field" % pid],
        ]
        features.append(tf.feature_column.input_layer(parsed_features, columns))
    context_embedding_mtx = tf.stack([
        tf.zeros_like(features[0], dtype=tf.float32),
        features[0],
        features[1],
        features[2]
    ])
    context_embedding_mtx = tf.transpose(context_embedding_mtx, perm=[1, 0, 2])
    context_embedding_mtx = tf.reshape(context_embedding_mtx, [-1, tf.shape(context_embedding_mtx)[2]])
    return context_embedding_mtx


def get_feature_seq_embedding(features):
    context_embedding_mtx = get_context_embedding_mtx(features)
    feature_seq = features["feature_seq"]
    dense_feature_seq = tf.sparse.to_dense(feature_seq)
    multiplier = tf.range(0, feature_seq.dense_shape[0]) * 4
    mask = tf.to_int64(tf.not_equal(dense_feature_seq, 0))
    feature_seq = dense_feature_seq + mask * tf.expand_dims(multiplier, axis=1)
    context_seq_embedding = tf.nn.embedding_lookup(context_embedding_mtx, feature_seq)
    context_seq_embedding.set_shape([None, None, 32])
    return context_seq_embedding
