import torch
from torch import nn

from .input_feature import InputFeature

# Mirrors the threat indexing of the data loader
# (data_loader/cpp/training_data_loader.cpp, FullThreats::threat_index) and
# Stockfish (src/nnue/features/full_threats.cpp). Piece types are
# 0..5 = pawn, knight, bishop, rook, queen, king; a threat piece id is
# color * 5 + type, since kings never attack or get attacked.
_PAWN, _KNIGHT, _BISHOP, _ROOK, _QUEEN = range(5)
_NUM_VALID_TARGETS = [4, 10, 8, 8, 10, 0]
# fmt: off
_MAP = [
    [-1,  0, -1,  1, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3, -1, -1],
    [ 0,  1,  2,  3,  4, -1],
    [-1, -1, -1, -1, -1, -1],
]
# fmt: on

_KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _ray_attacks(sq: int, dirs: list[tuple[int, int]], slide: bool) -> list[int]:
    f0, r0 = sq & 7, sq >> 3
    out = []
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            out.append(r * 8 + f)
            if not slide:
                break
            f, r = f + df, r + dr
    return out


def _attacks(piece_type: int, color: int, sq: int) -> list[int]:
    """Empty-board attacks, as bb::pawnAttacks and pseudoAttacks in the loader."""
    if piece_type == _PAWN:
        dr = 1 if color == 0 else -1
        return _ray_attacks(sq, [(-1, dr), (1, dr)], slide=False)
    if piece_type == _KNIGHT:
        return _ray_attacks(sq, _KNIGHT_DIRS, slide=False)
    if piece_type == _BISHOP:
        return _ray_attacks(sq, _BISHOP_DIRS, slide=True)
    if piece_type == _ROOK:
        return _ray_attacks(sq, _ROOK_DIRS, slide=True)
    if piece_type == _QUEEN:
        return _ray_attacks(sq, _BISHOP_DIRS + _ROOK_DIRS, slide=True)
    raise ValueError(piece_type)


def threat_feature_table() -> list[tuple[int, int, int, int, int]]:
    """Return (attacker, from, to, attacked, index) for every usable Full_Threats index.

    Squares and piece ids (color * 5 + type) are in the perspective's
    orientation, as the loader computes them. The index space reserves both
    directions for same-type pairs but uses only one, so 53,312 of the 59,808
    indices occur. Raises if two threats share an index or the total differs.
    """
    piece_offset = []  # threatoffsets[piece][65]
    square_offset = []  # threatoffsets[piece][from]
    total_squares = []  # threatoffsets[piece][64]
    offset = 0
    for piece in range(10):
        color, pt = divmod(piece, 5)
        piece_offset.append(offset)
        per_from = []
        acc = 0
        for sq in range(64):
            per_from.append(acc)
            if pt != _PAWN or 8 <= sq <= 55:
                acc += len(_attacks(pt, color, sq))
        square_offset.append(per_from)
        total_squares.append(acc)
        offset += _NUM_VALID_TARGETS[pt] * acc

    entries = []
    for attacker in range(10):
        ca, ta = divmod(attacker, 5)
        for frm in range(64):
            if ta == _PAWN and not 8 <= frm <= 55:
                continue
            targets = _attacks(ta, ca, frm)
            for to in targets:
                below = sum(1 for t in targets if t < to)
                for attacked in range(10):
                    cb, tb = divmod(attacked, 5)
                    slot = _MAP[ta][tb]
                    if slot < 0:
                        continue
                    enemy = ca != cb
                    if ta == tb and (enemy or ta != _PAWN) and frm < to:
                        continue
                    index = (
                        piece_offset[attacker]
                        + (cb * (_NUM_VALID_TARGETS[ta] // 2) + slot)
                        * total_squares[attacker]
                        + square_offset[attacker][frm]
                        + below
                    )
                    entries.append((attacker, frm, to, attacked, index))

    indices = {e[4] for e in entries}
    if (
        offset != FullThreats.NUM_INPUTS
        or len(indices) != len(entries)
        or not indices <= set(range(offset))
    ):
        raise RuntimeError("Full_Threats index table doesn't match the loader's")
    return entries


class FullThreats(InputFeature):
    HASH = 0x2C41EAEA
    FEATURE_NAME = "Full_Threats"
    INPUT_FEATURE_NAME = "Full_Threats"
    MAX_ACTIVE_FEATURES = 128

    NUM_INPUTS = 59808
    NUM_REAL_FEATURES = 59808
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)


class FullThreatsFactorized(FullThreats):
    """Full_Threats with a training-only factorizer.

    Each real feature (attacker, from, to, attacked) also uses a shared virtual
    row for (attacker, attacked, to), so all origin squares of the same threat
    on the same target square share a component. The virtual rows are added to
    the real ones at export, so the network file, hash and engine code are the
    same as for Full_Threats.
    """

    FEATURE_NAME = "Full_Threats^"
    NUM_THREAT_PIECES = 10
    # One row per (attacker, attacked, to), plus a zero row for the indices
    # that never occur.
    UNUSED_ROW = NUM_THREAT_PIECES * NUM_THREAT_PIECES * 64  # 6400
    NUM_INPUTS_VIRTUAL = UNUSED_ROW + 1
    _CLIP_CHUNK_ROWS = 8192

    def __init__(self, num_outputs: int):
        super().__init__(num_outputs)

        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        virtual_index = torch.full(
            (self.NUM_INPUTS,), self.UNUSED_ROW, dtype=torch.long
        )
        for attacker, _frm, to, attacked, index in threat_feature_table():
            virtual_index[index] = (
                attacker * self.NUM_THREAT_PIECES + attacked
            ) * 64 + to
        # Derived from the indexing, so it isn't saved in checkpoints.
        self.register_buffer("virtual_index", virtual_index, persistent=False)

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.virtual_weight.index_select(0, self.virtual_index)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.add_(self.virtual_weight.index_select(0, self.virtual_index))
        self.zero_virtual_weights()

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self) -> None:
        self.zero_virtual_weights()

    @torch.no_grad()
    def clip_weights(self, quantization) -> None:
        # Clip the merged weight to the export range and keep the virtual part,
        # in chunks to avoid a full-size temporary.
        min_w, max_w = quantization.get_ft_weight_bounds(self.EXPORT_WEIGHT_DTYPE)
        w = self.weight.data
        for start in range(0, self.NUM_INPUTS, self._CLIP_CHUNK_ROWS):
            end = min(start + self._CLIP_CHUNK_ROWS, self.NUM_INPUTS)
            v = self.virtual_weight.data.index_select(0, self.virtual_index[start:end])
            w[start:end].add_(v).clamp_(min_w, max_w).sub_(v)

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.merged_weight().detach().clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)
        self.zero_virtual_weights()
