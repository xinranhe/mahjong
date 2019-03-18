import argparse
from multiprocessing import Pool
import os
import sys
import tensorflow as tf

from log_parser.discard_prediction_parser import parse_discard_prediction

NUM_THREADS = 12

MAX_SCORE_DIFF = 32000
SCORE_DIFF_DET = 400

CLS_TOKEN = 69
SEP_TOKEN = 70
PADDING = 0

INPUT_DATA_FOLDER = "data/raw"
OUTPUT_DATA_DIR = "data/tfrecord"
TFRECORD_OPTION = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def get_center_features(one_round):
    global_context = one_round.global_context
    center_context = one_round.center_player.context
    feature_dict = {}
    feature_dict["current_field"] = _int64_feature(global_context.field)
    feature_dict["round"] = _int64_feature(global_context.round)
    feature_dict["center_field"] = _int64_feature(center_context.field)
    feature_dict["center_oya"] = _int64_feature(int(center_context.is_dealer))
    return feature_dict

def is_player_riichi(player):
    for discarded_hai in player.discarded_hai:
        if discarded_hai.is_after_riichi:
            return True
    return False

def player_score_diff(center_score, player_score):
    diff = max(min(MAX_SCORE_DIFF, center_score - player_score), -MAX_SCORE_DIFF)
    diff = diff // SCORE_DIFF_DET
    return diff + MAX_SCORE_DIFF / SCORE_DIFF_DET

def get_player_features(one_round):
    feature_dict = {}
    center_context = one_round.center_player.context
    for pid, player in enumerate(one_round.other_player):
        player_context = player.context
        feature_dict["player%d_oya" % pid] = _int64_feature(player_context.is_dealer)
        feature_dict["player%d_field" % pid] = _int64_feature(player_context.field)
        feature_dict["player%d_riichi" % pid] = _int64_feature(int(is_player_riichi(player)))
        feature_dict["player%d_claim" % pid] = _int64_feature(len(player.claim))
        feature_dict["player%d_order" % pid] = _int64_feature(3 + center_context.order - player_context.order)
        feature_dict["player%d_score" % pid] = _int64_feature(player_score_diff(center_context.score, player_context.score))
    return feature_dict


def get_hid(hai, doras):
    if hai.id in doras or hai.is_red:
        return 2 * (hai.id + 1)
    else:
        return hai.id + 1   

def get_sequence_features(one_round):
    center_player = one_round.center_player
    doras = [h.id for h in one_round.global_context.dora]
    
    hai_seq = [CLS_TOKEN]
    pos_seq = [0] * 15
    feature_seq = [0] * 15
    
    hai_seq.extend([get_hid(h, doras) for h in center_player.hand])
    if len(center_player.hand) < 14:
        hai_seq.extend([PADDING] * (14 - len(center_player.hand)))
    for i, player in enumerate(one_round.other_player):
        hai_seq.append(SEP_TOKEN)
        pos_seq.append(0)
        feature_seq.append(0)
        
        hai_seq.extend([get_hid(h.hai, doras) for h in player.discarded_hai])
        pos_seq.extend(range(1, 1 + len(player.discarded_hai)))
        feature_seq.extend([i + 1] * len(player.discarded_hai))
    return {
        "hai_seq": _int64_list_feature(hai_seq),
        "pos_seq": _int64_list_feature(pos_seq),
        "feature_seq": _int64_list_feature(feature_seq)
    }

def get_discard_label(one_round):
    label = [0.0] * 14
    discarded_hai = one_round.discarded_hai
    count = 0
    for i, hai in enumerate(one_round.center_player.hand):
        if hai.id == discarded_hai.id:
            count += 1
            label[i] = 1
    for i in xrange(14):
            label[i] /= count
    return {"label": _float_list_feature(label)}

def generate_tfexample(one_round):
    features = {}
    features.update(get_center_features(one_round))
    features.update(get_player_features(one_round))
    features.update(get_sequence_features(one_round))
    features.update(get_discard_label(one_round))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def generate_data(folder):
    folder_path = "%s/%s" % (INPUT_DATA_FOLDER, folder)
    writer = tf.python_io.TFRecordWriter("%s/%s.gz" % (OUTPUT_DATA_DIR, folder), options=TFRECORD_OPTION)
    num_tfrecords = 0
    num_failed_files = 0
    for i, file in enumerate(os.listdir(folder_path)):
        print "processed %d files with %d failed: %d tfrecords" % (i, num_failed_files, num_tfrecords)
        file_path = "%s/%s" % (folder_path, file)
        try:
            games = parse_discard_prediction(open(file_path, "r").read())
            for game in games:
                for one_round in game.one_round:
                    center_player = one_round.center_player
                    # skip discard hai after riichi or player offline
                    if center_player.is_riichi or center_player.is_offline:
                        continue
                    writer.write(generate_tfexample(one_round).SerializeToString())
                    num_tfrecords += 1
        except:
            num_failed_files += 1
            print "Failed in parseing:", file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--start_date', default='')
    parser.add_argument('--end_date', default='')
    known_args, _ = parser.parse_known_args(sys.argv)

    date_to_process = []
    for date in os.listdir(INPUT_DATA_FOLDER):
        if date >= known_args.start_date and date <= known_args.end_date:
            date_to_process.append(date)

    print date_to_process
    # multithread generate training data
    p = Pool(NUM_THREADS)
    p.map(generate_data, date_to_process)

