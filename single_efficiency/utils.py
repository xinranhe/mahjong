#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mahjong.shanten import Shanten
from mahjong.agari import Agari
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.tile import TilesConverter

SHANTEN = Shanten()
CALCULATOR = HandCalculator()
AGARI = Agari()


TO_GRAPH_LIST = [
    "🀇", "🀈", "🀉", "🀊", "🀋", "🀌", "🀍", "🀎", "🀏", "🀙", "🀚", "🀛", "🀜", "🀝", "🀞", "🀟", "🀠", "🀡",
    "🀐", "🀑", "🀒", "🀓", "🀔", "🀕", "🀖", "🀗", "🀘", "🀀", "🀁", "🀂", "🀃", "🀆", "🀅", "🀄"
]
NUM_HAIS = 34

def get_total_score(hand34, wintile34):
    result = CALCULATOR.estimate_hand_value(TilesConverter.to_136_array(hand34), wintile34 * 4, config=HandConfig(is_tsumo=True))
    return result.cost['main'] + 2 * result.cost['additional']


def is_agari(hand34):
    return AGARI.is_agari(hand34)


def tiles34_to_list(tiles):
    result = []
    for i in xrange(34):
        for j in xrange(tiles[i]):
            result.append(i)
    return sorted(result)


def print_tile34_hand(tiles):
    result = ""
    for i in xrange(NUM_HAIS):
        for j in xrange(tiles[i]):
            result += TO_GRAPH_LIST[i]
    print result


def load_hand(files):
    hands = []
    for file_path in files:
        for line in open(file_path, "r"):
            fields = line[:-1].split(":")
            shanten = int(fields[0])
            hand = [int(hid) for hid in fields[1].split(",")]
            hands.append((shanten, hand))
    return hands
