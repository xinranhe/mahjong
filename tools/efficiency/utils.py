#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mahjong.tile import TilesConverter
from mahjong.shanten import Shanten
import numpy as np

SHANTEN = Shanten()
NUM_HAIS = 34
TO_GRAPH_LIST = [
    "🀇", "🀈", "🀉", "🀊", "🀋", "🀌", "🀍", "🀎", "🀏", "🀙", "🀚", "🀛", "🀜", "🀝", "🀞", "🀟", "🀠", "🀡",
    "🀐", "🀑", "🀒", "🀓", "🀔", "🀕", "🀖", "🀗", "🀘", "🀀", "🀁", "🀂", "🀃", "🀆", "🀅", "🀄"
]

HAI_NAMES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "don", "nan", "xia", "pei", "bai", "zhong", "fa"
];

# len(used_tiles) % 2 == 1
def compute_jinzhang(tiles, used_tiles):
    initial_shanten = SHANTEN.calculate_shanten(tiles)
    result = np.zeros(NUM_HAIS)
    for i in xrange(NUM_HAIS):
        num_hais = 4 - used_tiles[i] - tiles[i]
        if num_hais > 0:
            tiles[i] += 1
            new_shanten = SHANTEN.calculate_shanten(tiles)
            if (new_shanten < initial_shanten):
                result[i] = num_hais
            tiles[i] -= 1
    return np.sum(result), result


def compute_discard_jinzhange(tiles, used_tiles):
    initial_shanten = SHANTEN.calculate_shanten(tiles)
    results = []
    best_jinzhang_num = 0
    for i in xrange(NUM_HAIS):
        if tiles[i] > 0:
            tiles[i] -= 1
            new_shanten = SHANTEN.calculate_shanten(tiles)
            if new_shanten == initial_shanten:
                num_jinzhang, _ = compute_jinzhang(tiles, used_tiles)
                if num_jinzhang > best_jinzhang_num:
                    best_jinzhang_num = num_jinzhang
                results.append((num_jinzhang, i))
            tiles[i] += 1
    discards = [r[1] for r in results if r[0] == best_jinzhang_num]
    return (best_jinzhang_num, discards)


def compute_discard_all(tiles, used_tiles):
    results = []
    for i in xrange(NUM_HAIS):
        if tiles[i] > 0:
            tiles[i] -= 1
            new_shanten = SHANTEN.calculate_shanten(tiles)
            num_jinzhang, _ = compute_jinzhang(tiles, used_tiles)
            results.append((i, new_shanten, num_jinzhang))
            tiles[i] += 1
    results = sorted(results, key=lambda x: (x[1], -x[2], x[0]))
    return results


# len(used_tiles) % 2 == 1
def compute_hand(tiles, used_tiles):
    initial_shanten = SHANTEN.calculate_shanten(tiles)
    num_initial_jinzhang, _ = compute_jinzhang(tiles, used_tiles)
    all_jinzhang = []
    all_gailiang = []
    for i in xrange(NUM_HAIS):
        tiles[i] += 1
        used_tiles[i] += 1
        new_shanten = SHANTEN.calculate_shanten(tiles)
        # jinzhang
        if new_shanten <= initial_shanten:
            best_jinzhang_num, discards = compute_discard_jinzhange(tiles, used_tiles)
            if new_shanten < initial_shanten:
                all_jinzhang.append((i, best_jinzhang_num, discards))
            else:
                if best_jinzhang_num >= num_initial_jinzhang + 4:
                    all_gailiang.append((i, best_jinzhang_num - num_initial_jinzhang, discards))
        tiles[i] -= 1
        used_tiles[i] -= 1
    all_gailiang = sorted(all_gailiang, key=lambda x: (-x[1], x[0]))
    return all_jinzhang, all_gailiang


# len(used_tiles) % 2 == 0
def generate_discard_html(tiles, used_tiles):
    results = compute_discard_all(tiles, used_tiles)
    result = "<h2>%d向听</h2><br>" % (SHANTEN.calculate_shanten(tiles))
    result += '<table border="1">'
    result += "<tr><th>舍牌</th><th>向听数</th><th>进张数</th></tr>"    
    for (i, new_shanten, num_jinzhang) in results:
        result += "<tr>"
        result += '<th><img src="data/%s.png" width="50" height="80" onclick="exchange(-1, %d)"></th>' % (HAI_NAMES[i], i)
        result += '<th>%d</th>' % new_shanten
        result += '<th>%d</th>' % num_jinzhang
        result += "<tr/>"
    result += "</table>"
    return result


def generate_hand_html(tiles, used_tiles):
    all_jinzhang, all_gailiang = compute_hand(tiles, used_tiles)
    def generate_table(input):
        result = '<table border="1">'
        result += "<tr><th>牌</th><th>舍牌</th><th>进张数</th></tr>"        
        for (i, best_jinzhang_num, discards) in input:
            result += "<tr>"
            result += '<th><img src="data/%s.png" width="50" height="80" onclick="exchange(%d, -1)"></th>' % (HAI_NAMES[i], i)
            result += '<th>'
            for hid in discards:
                result += '<img src="data/%s.png" width="50" height="80" onclick="exchange(%d, %d)">' % (HAI_NAMES[hid], i, hid)
            result += '</th>'
            result += '<th>%d</th>' % best_jinzhang_num
            result += "<tr/>"
        result += "</table>"
        return result

    num_jinzhang = 0
    for r in all_jinzhang:
        num_jinzhang += 4 - used_tiles[r[0]] - tiles[r[0]]
    result = "<h2>%d向听</h2>" % SHANTEN.calculate_shanten(tiles)
    result += "<h2>%d进张</h2>" % num_jinzhang
    result += generate_table(all_jinzhang)
    result += "<h2>改良</h2>"
    result += generate_table(all_gailiang)
    return result

if __name__ == '__main__':
    tiles = TilesConverter.string_to_34_array(man='', 
                                              pin='234445789', 
                                              sou='1568')
    used_tiles = [0] * NUM_HAIS
    all_jinzhang, all_gailiang = compute_hand(tiles, used_tiles)
    for (i, best_jinzhang_num, discards) in all_jinzhang:
        print "%s: %d -> %s" % (TO_GRAPH_LIST[i], best_jinzhang_num, ",".join([TO_GRAPH_LIST[d] for d in discards]))
    print generate_hand_html((all_jinzhang, all_gailiang))
