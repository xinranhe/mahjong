from gen_file.proto import discard_prediction_data_pb2
from log_parser.xml_parser import parse_mjlog
from log_parser.utils import load_mjlog

HAI_CHARS = [u"1m", u"2m", u"3m", u"4m", u"5m", u"6m", u"7m", u"8m", u"9m",
             u"1p", u"2p", u"3p", u"4p", u"5p", u"6p", u"7p", u"8p", u"9p",
             u"1s", u"2s", u"3s", u"4s", u"5s", u"6s", u"7s", u"8s", u"9s",
             u"don", u"nan", u"xia", u"pei", u"bai", u"fa", u"zhong"]

def get_player_info(meta_node, field_round):
    player_info = {}
    for i, data in enumerate(meta_node["UN"]):
        player_context = discard_prediction_data_pb2.PlayerContext()
        player_context.rate = data["rate"]
        player_context.dan = data["dan"]
        player_context.field = (i - field_round + 4) % 4
        player_info[i] = player_context
    return player_info

def get_init_player(data):
    players = {}
    for i in xrange(4):
        players[i] = {}
        players[i]["discard"] = []
        players[i]["field"] = i
        players[i]["is_riichi"] = False
        players[i]["is_offline"] = False
        players[i]["claim"] = []
    # initial hands
    for i, hands in enumerate(data["hands"]):
        players[i]["hand"] = set(hands)
    # scores
    for i, score in enumerate(data["scores"]):
        players[i]["score"] = score
    return players

def get_hai(hid):
    hai = discard_prediction_data_pb2.Hai()
    hai.id = hid >> 2
    hai.char = HAI_CHARS[hai.id]
    hai.is_red = (hid in [16,52,88])
    return hai

def get_claim(claim):
    claim_proto = discard_prediction_data_pb2.Claim()
    claim_proto.hai.extend([get_hai(hid) for hid in claim])
    return claim_proto

def get_discard_hai(hid, player):
    discard_hai = discard_prediction_data_pb2.DiscardedHai()
    discard_hai.hai.MergeFrom(get_hai(hid))
    discard_hai.num_claim = len(player["claim"])
    discard_hai.is_after_riichi = player["is_riichi"]
    return discard_hai

def parse_discard_prediction(xml_string):
    root_node = parse_mjlog(load_mjlog(xml_string))
    games = []
    for match in root_node["rounds"]:
        results = []
        for round_node in match:
            # new game
            tag = round_node["tag"]
            data = round_node["data"]
            if tag == "INIT":
                pos = 0
                combo =  data["combo"]
                field = (data['round'] / 4) % 4
                field_round = data['round'] % 4
                oya = int(data["oya"])
                doras = [get_hai(data["dora"])]
                players = get_init_player(data)
                player_info = get_player_info(root_node["meta"], field_round)
            elif tag == "DORA":
                doras.append(get_hai(data["hai"]))
            elif tag == "DRAW":
                players[data["player"]]["hand"].add(data["tile"])
            elif tag == "CALL":
                # handle kakan
                if data["call_type"] == "KaKan":
                    hid = data["mentsu"][0] >> 2
                    players[data["caller"]]["claim"] = list(filter(lambda c: hid not in [i >> 2 for i in c], players[data["caller"]]["claim"])) 
                players[data["caller"]]["claim"].append(data["mentsu"])
                for hid in data["mentsu"]:
                    if hid in players[data["caller"]]["hand"]:
                        players[data["caller"]]["hand"].remove(hid)
            elif tag == "BYE":
                players[data["index"]]["is_offline"] = True
            elif tag == "RESUME":
                players[data["index"]]["is_offline"] = False
            elif tag == "REACH":
                if data["step"] == 1:
                    players[data["player"]]["is_riichi"] = True
                elif data["step"] == 2:
                    for i, score in enumerate(data["scores"]):
                        players[i]["score"] = score
            elif tag == "DISCARD":
                current_player = data["player"]
                # global copntext
                global_context = discard_prediction_data_pb2.GlobalContext()
                global_context.field = field
                global_context.round = field_round
                global_context.combo = combo
                global_context.turn = pos
                global_context.dora.extend(doras)

                def get_player_context(pid):
                    player_context = player_info[pid]
                    player_context.is_dealer = (oya == pid)
                    player_context.score = players[pid]["score"]
                    order = 1
                    for i in xrange(4):
                        if i == pid:
                            continue
                        if players[i]["score"] >= player_context.score:
                            order += 1
                    player_context.order = order
                    return player_context

                # Center player
                center_player = discard_prediction_data_pb2.CenterPlayer()
                center_player.context.MergeFrom(get_player_context(current_player))
                center_player.claim.extend([get_claim(c) for c in players[current_player]["claim"]])
                center_player.hand.extend([get_hai(hid) for hid in sorted(players[current_player]["hand"])])
                center_player.is_riichi = players[current_player]["is_riichi"]
                center_player.is_offline = players[current_player]["is_offline"]

                # other player
                def get_other_player(pid):
                    other_player = discard_prediction_data_pb2.OtherPlayer()
                    other_player.context.MergeFrom(get_player_context(pid))
                    other_player.claim.extend([get_claim(c) for c in players[pid]["claim"]])
                    other_player.discarded_hai.extend(players[pid]["discard"])
                    return other_player
                other_players = []
                for i in xrange(4):
                    if i == current_player:
                        continue
                    other_players.append(get_other_player(i))

                # form proto
                one_round = discard_prediction_data_pb2.OneRound()
                one_round.global_context.MergeFrom(global_context)
                one_round.center_player.MergeFrom(center_player)
                one_round.other_player.extend(other_players)
                one_round.discarded_hai.MergeFrom(get_hai(data["tile"]))
                results.append(one_round)

                # handle discard
                players[current_player]["discard"].append(get_discard_hai(data["tile"], players[current_player]))
                players[current_player]["hand"].remove(data["tile"])
                pos += 1
        
        game = discard_prediction_data_pb2.Game()
        game.one_round.extend(results)
        games.append(game)

    return games
