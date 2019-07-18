import numpy as np
import random

from single_efficiency import utils

class GreedyJinZhang(object):
    def discard(self, tiles, left_tiles):
        initial_shanten = utils.SHANTEN.calculate_shanten(tiles)
        results = []
        best_jinzhang_num = 0
        for i in xrange(utils.NUM_HAIS):
            if tiles[i] > 0:
                tiles[i] -= 1
                new_shanten = utils.SHANTEN.calculate_shanten(tiles)
                if new_shanten == initial_shanten:
                    num_jinzhang = self.compute_jinzhang(tiles, left_tiles)
                    if num_jinzhang > best_jinzhang_num:
                        best_jinzhang_num = num_jinzhang
                    results.append((num_jinzhang, i))
                tiles[i] += 1
        discards = [r[1] for r in results if r[0] == best_jinzhang_num]
        return random.choice(discards)

    def compute_jinzhang(self, hand34, left_tiles):
        initial_shanten = utils.SHANTEN.calculate_shanten(hand34)
        result = np.zeros(utils.NUM_HAIS)
        for i in xrange(utils.NUM_HAIS):
            num_hais = left_tiles[i]
            if num_hais > 0:
                hand34[i] += 1
                new_shanten = utils.SHANTEN.calculate_shanten(hand34)
                if (new_shanten < initial_shanten):
                    result[i] = num_hais
                hand34[i] -= 1
        return np.sum(result)


class GreedyShanten(object):
    def discard(self, tiles, left_tiles):
        initial_shanten = utils.SHANTEN.calculate_shanten(tiles)
        discards = []
        for i in xrange(34):
            if tiles[i] > 0:
                tiles[i] -= 1
                if utils.SHANTEN.calculate_shanten(tiles) == initial_shanten:
                    discards.append(i)
                tiles[i] += 1
        return random.choice(discards)


class EpislonGreedy(object):
    def __init__(self, strategy, epsilon):
        self.strategy = strategy
        self.epsilon = epsilon

    def discard(self, tiles, left_tiles):
        hands = []
        possible_tiles = []
        for i in xrange(34):
            if tiles[i] > 0:
                tiles[i] -= 1
                hands.append(utils.tiles34_to_list(tiles))
                possible_tiles.append(i)
                tiles[i] += 1
        if np.random.uniform() <= self.epsilon:
            return random.choice(possible_tiles)
        else:
            return self.strategy.discard(tiles, left_tiles)
