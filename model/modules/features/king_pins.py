from .pp_3wide import PP3Wide


class KingPins(PP3Wide):
    """An enemy slider aimed at a king with exactly one non-king piece between them.

    Other enemy sliders on the king's lines are ignored when counting. Indexed by
    whose king it is, the slider type, the piece in between and its square. Must
    match Features::KingPins in Stockfish and KingPins in the C++ data loader.
    """

    HASH = 0x7B1E4C93
    FEATURE_NAME = "King_Pins"
    INPUT_FEATURE_NAME = "King_Pins"
    MAX_ACTIVE_FEATURES = 30

    NUM_INPUTS = 3840
    NUM_REAL_FEATURES = 3840
