import random

from single_efficiency import utils

MAX_ROUND = 60
NEGATIVE_REWARD = 0
DISCOUNT_FACTOR = 0.99

class Strategy(object):
    # both tiles, left_tiles in tile34 format
    # tiles contains 14 tiles
    # should return tile34 id to discard
    def discard(self, tiles, left_tiles):
        pass


def simulate(init_hand, strategy, is_debug=False):
    # init
    current_hand = [0] * utils.NUM_HAIS
    left_tiles = [4] * utils.NUM_HAIS
    for hai in init_hand:
        left_tiles[hai] -= 1
        current_hand[hai] += 1
    yama = utils.tiles34_to_list(left_tiles)
    random.shuffle(yama)
    
    rewards = [-1] * MAX_ROUND
    # (hand, reward)
    data = []
    for i in xrange(MAX_ROUND):
        data.append([utils.tiles34_to_list(current_hand), NEGATIVE_REWARD])
        
        # draw new tile
        new_tile = yama[i]
        
        if is_debug:
            utils.print_tile34_hand(current_hand)
            print "Draw:" + utils.TO_GRAPH_LIST[new_tile]
        
        is_agari = False
        current_hand[new_tile] += 1
        left_tiles[new_tile] -= 1
        if utils.is_agari(current_hand):
            data[i][1] = utils.get_total_score(current_hand, new_tile) / 100.0
            is_agari = True
            if is_debug:
                print "Agali score: %f hand:" % data[i][1],
                utils.print_tile34_hand(current_hand)
                print "============"
            break
        else:
            discard = strategy.discard(current_hand, left_tiles)
            current_hand[discard] -= 1
            if is_debug:
                print "Discard:" + utils.TO_GRAPH_LIST[discard]
    
    if not is_agari:
        return []
    # generate reward
    i -= 1
    while i >= 0:
        data[i][1] += data[i + 1][1] * DISCOUNT_FACTOR
        i -= 1
    
    return data
