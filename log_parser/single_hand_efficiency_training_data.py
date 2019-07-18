import argparse
from mahjong.shanten import Shanten
from multiprocessing import Pool
import os
import sys

from log_parser.discard_prediction_parser import parse_discard_prediction

SHANTEN = Shanten()

INPUT_DATA_FOLDER = "data/raw"
OUTPUT_DATA_DIR = "data/single_hand_efficiency"

def tiles34_to_list(tiles):
    result = []
    for i in xrange(34):
        for j in xrange(tiles[i]):
            result.append(i)
    return sorted(result)


def generate_data(folder):
    folder_path = "%s/%s" % (INPUT_DATA_FOLDER, folder)
    writer = open("%s/%s.txt" % (OUTPUT_DATA_DIR, folder), "w")
    num_hands = [0] * 7
    num_failed_files = 0
    for i, file in enumerate(os.listdir(folder_path)):
        print "processed %d files with %d failed: %s records" % (i, num_failed_files, ",".join([str(n) for n in num_hands]))
        file_path = "%s/%s" % (folder_path, file)
        try:
            games = parse_discard_prediction(open(file_path, "r").read())
            for game in games:
                for one_round in game.one_round:
                    hais = one_round.center_player.hand
                    if len(hais) != 14:
                        continue
                    hand = [0] * 34
                    for hai in hais:
                        hand[hai.id] += 1
                    if hand[one_round.discarded_hai.id] <= 0:
                        continue
                    hand[one_round.discarded_hai.id] -= 1
                    shanten = int(SHANTEN.calculate_shanten(hand))
                    num_hands[shanten] += 1
                    writer.write("%d:%s\n" % (shanten, ",".join([str(i) for i in tiles34_to_list(hand)])))
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
    generate_data(date_to_process[0])
    # multithread generate training data
    #p = Pool(NUM_THREADS)
    #p.map(generate_data, date_to_process)
