import random
import tensorflow as tf

BUFFER_SIZE = 10000
NUM_PARSING_THREADS = 8
NUM_PARALLEL_READERS = 1
PREFETCH_BUFFER_SIZE = 1

def get_parse_spec():
    parse_spec = {
        "hai_seq": tf.VarLenFeature(tf.int64),
        "pos_seq": tf.VarLenFeature(tf.int64),
        "feature_seq": tf.VarLenFeature(tf.int64),
        "label": tf.FixedLenFeature([14], tf.float32),
        "is_riichi": tf.FixedLenFeature([1], tf.int64),
        "is_anyone_riichi": tf.FixedLenFeature([1], tf.int64),
    }
    parse_spec["field"] = tf.FixedLenFeature([1], tf.int64)
    parse_spec["round"] = tf.FixedLenFeature([1], tf.int64)
    parse_spec["turn"] = tf.FixedLenFeature([1], tf.int64)
    for name in ["center", "player0", "player1", "player2"]:
        parse_spec["%s_oya" % name] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["%s_claim" % name] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["%s_order" % name] = tf.FixedLenFeature([1], tf.int64)
        if "player" in name:
            parse_spec["%s_riichi" % name] = tf.FixedLenFeature([1], tf.int64)
    return parse_spec


def _tfrecord_parse_fn(example_proto):
    parsed_features = tf.parse_single_example(example_proto, get_parse_spec())
    labels = {
        "label": parsed_features["label"],
        "is_riichi": tf.cast(parsed_features["is_riichi"], dtype=tf.float32),
    }
    return parsed_features, labels


def input_function(filename_patterns, is_train, parameters):
    parse_spec = get_parse_spec()

    def input_fn():
        input_files = []
        for filename_pattern in filename_patterns:
            input_files.extend(tf.gfile.Glob(filename_pattern))
        if is_train:
            random.shuffle(input_files)
        files = tf.data.Dataset.list_files(input_files)
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            lambda fs: tf.data.TFRecordDataset(fs, compression_type=parameters["compression_type"]), cycle_length=parameters["num_parallel_readers"]))
        if is_train:
            dataset = dataset.shuffle(parameters["buffer_size"])
        dataset = dataset.map(_tfrecord_parse_fn, num_parallel_calls=parameters["num_parsing_threads"])
        dataset = dataset.batch(parameters["batch_size"])
        dataset = dataset.prefetch(buffer_size=parameters["prefetch_buffer_size"])
        dataset = dataset.repeat()
        return dataset
    return input_fn
