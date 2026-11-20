"""
Atari-57 Human and Random Baseline Normalization Utilities.
Values sourced from DeepMind Nature benchmark (Mnih et al., Nature 2015)
and Google Dopamine (Castro et al., 2018).
"""

# Atari-57 Random and Human baselines from Mnih et al. (2015) & Badia et al. (2020)
ATARI_RANDOM_SCORES = {
    "alien": 227.8,
    "amidar": 5.8,
    "assault": 222.4,
    "asterix": 210.0,
    "asteroids": 719.1,
    "atlantis": 12850.0,
    "bank_heist": 14.2,
    "battle_zone": 2360.0,
    "beam_rider": 363.9,
    "berzerk": 123.7,
    "bowling": 23.1,
    "boxing": 0.1,
    "breakout": 1.7,
    "carnival": 687.6,
    "centipede": 2090.9,
    "chopper_command": 811.0,
    "crazy_climber": 10780.5,
    "defender": 1965.5,
    "demon_attack": 152.1,
    "double_dunk": -18.6,
    "enduro": 0.0,
    "fishing_derby": -91.7,
    "freeway": 0.0,
    "frostbite": 65.2,
    "gopher": 257.6,
    "gravitar": 173.0,
    "hero": 1027.0,
    "ice_hockey": -11.2,
    "jamesbond": 29.0,
    "kangaroo": 52.0,
    "krull": 1598.0,
    "kung_fu_master": 258.5,
    "montezuma_revenge": 0.0,
    "ms_pacman": 307.3,
    "mspacman": 307.3,
    "name_this_game": 2292.3,
    "phoenix": 761.4,
    "pitfall": -229.4,
    "pong": -20.7,
    "pooyan": 86.0,
    "private_eye": 24.9,
    "qbert": 163.9,
    "riverraid": 1338.5,
    "road_runner": 11.5,
    "robotank": 2.2,
    "seaquest": 68.4,
    "skiing": -17098.1,
    "solaris": 1236.3,
    "space_invaders": 148.0,
    "spaceinvaders": 148.0,
    "star_gunner": 664.0,
    "surround": -10.0,
    "tennis": -23.8,
    "time_pilot": 3568.0,
    "tutankham": 11.4,
    "up_n_down": 533.4,
    "venture": 0.0,
    "video_pinball": 16256.9,
    "wizard_of_wor": 563.5,
    "yars_revenge": 3092.9,
    "zaxxon": 32.5,
}

ATARI_HUMAN_SCORES = {
    "alien": 7127.7,
    "amidar": 1719.5,
    "assault": 742.0,
    "asterix": 8503.3,
    "asteroids": 47388.7,
    "atlantis": 29028.1,
    "bank_heist": 753.1,
    "battle_zone": 37187.5,
    "beam_rider": 16926.5,
    "berzerk": 2630.4,
    "bowling": 160.7,
    "boxing": 12.1,
    "breakout": 30.5,
    "carnival": 4030.0,
    "centipede": 12017.0,
    "chopper_command": 7387.8,
    "crazy_climber": 35829.4,
    "defender": 18688.9,
    "demon_attack": 1971.0,
    "double_dunk": -16.4,
    "enduro": 860.5,
    "fishing_derby": -38.7,
    "freeway": 29.6,
    "frostbite": 4334.7,
    "gopher": 2412.5,
    "gravitar": 3351.4,
    "hero": 30826.4,
    "ice_hockey": 0.9,
    "jamesbond": 302.8,
    "kangaroo": 3035.0,
    "krull": 2665.5,
    "kung_fu_master": 22736.3,
    "montezuma_revenge": 4753.3,
    "ms_pacman": 6951.6,
    "mspacman": 6951.6,
    "name_this_game": 8049.0,
    "phoenix": 7242.6,
    "pitfall": 6463.7,
    "pong": 14.6,
    "pooyan": 13254.4,
    "private_eye": 69571.3,
    "qbert": 13455.0,
    "riverraid": 17118.0,
    "road_runner": 7845.0,
    "robotank": 11.9,
    "seaquest": 42054.7,
    "skiing": -4336.9,
    "solaris": 12326.7,
    "space_invaders": 1668.7,
    "spaceinvaders": 1668.7,
    "star_gunner": 10250.0,
    "surround": 6.5,
    "tennis": -8.3,
    "time_pilot": 5229.2,
    "tutankham": 167.6,
    "up_n_down": 11693.2,
    "venture": 1187.5,
    "video_pinball": 17647.9,
    "wizard_of_wor": 4756.5,
    "yars_revenge": 54576.9,
    "zaxxon": 9173.3,
}


def canonicalize_game_name(game_name: str) -> str:
    """Standardizes names like ALE/Pong-v5 or PongNoFrameskip-v4 to 'pong'."""
    name = game_name.lower()
    if "/" in name:
        name = name.split("/")[-1]
    name = name.split("-")[0].replace("noframeskip", "").replace("_", "")
    return name


def compute_human_normalized_score(game_name: str, raw_score: float) -> float:
    """
    Computes human-normalized score:
      HNS = (Score - Random_Score) / (Human_Score - Random_Score)
      0.0 = Random Agent baseline
      1.0 = Human Player baseline
    """
    canon = canonicalize_game_name(game_name)
    r = ATARI_RANDOM_SCORES.get(canon)
    h = ATARI_HUMAN_SCORES.get(canon)

    if r is None or h is None:
        raise KeyError(f"Game '{game_name}' (canonical: '{canon}') not found in Atari baselines.")

    if h == r:
        return 0.0
    return (raw_score - r) / (h - r)

