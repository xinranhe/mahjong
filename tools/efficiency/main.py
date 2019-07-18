from flask import Flask
from flask import request
from flask_cors import CORS

from utils import compute_hand,generate_hand_html,generate_discard_html

app = Flask(__name__)
CORS(app)

@app.route('/compute')
def normalize_word():
    hais = request.args.get("hais")

    hai_ids = [int(s) for s in hais.split(",")]
    tiles = [0] * 34
    for i in hai_ids:
        tiles[i] += 1

    if len(hai_ids) in [13, 10, 7, 4, 1]:
        return generate_hand_html(tiles, [0] * 34), 200
    else:
        return generate_discard_html(tiles, [0] * 34), 200


if __name__ == "__main__":
    app.run()
