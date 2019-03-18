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
    }
    parse_spec["current_field"] = tf.FixedLenFeature([1], tf.int64)
    parse_spec["round"] = tf.FixedLenFeature([1], tf.int64)
    parse_spec["center_field"] = tf.FixedLenFeature([1], tf.int64)
    parse_spec["center_oya"] = tf.FixedLenFeature([1], tf.int64)
    for pid in xrange(3):
        parse_spec["player%d_oya" % pid] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["player%d_field" % pid] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["player%d_riichi" % pid] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["player%d_claim" % pid] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["player%d_order" % pid] = tf.FixedLenFeature([1], tf.int64)
        parse_spec["player%d_score" % pid] = tf.FixedLenFeature([1], tf.int64)
    return parse_spec


def _tfrecord_parse_fn(example_proto):
    parsed_features = tf.parse_single_example(example_proto, get_parse_spec())
    return parsed_features, parsed_features["label"]


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
