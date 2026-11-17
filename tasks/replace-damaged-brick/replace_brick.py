"""On the station table, 48 cm in front of the arms: a solid brick wall, two bricks thick, 60 cm long and six
courses (40 cm) high, its face toward the arms. Every brick and joint is set: nothing in the wall moves. In the
fourth course from the bottom, in the middle of the wall, one damaged brick has been cut out together with the
mortar around it, leaving a pocket 214 mm wide, 77 mm high and 92 mm deep. The back of the pocket is the
wall's inner layer. Two broken pieces of the old brick still lie in the pocket.

Left of the pocket, in front of the wall, two mason's boards (bankers), 17.5 cm high, each hold one brick, lying
on its bed with its long side toward the arms and overhanging the board's front edge by 4 cm. The nearer one is
194 x 92 x 57 mm and about 1.9 kg, the same size and colour as the wall's bricks; the farther one is the same colour
but longer: 292 x 92 x 57 mm (a Norman brick). Right of the wall stands a timber mortar board with a patch of fresh
mortar on its far end, and on its near end a small trowel, blade toward the wall, already loaded with mortar.
Behind the right arm stands a black debris bucket, 13 cm tall.

The pocket is finished when the broken pieces are in the bucket, the pocket's bed is covered with fresh mortar,
and then the wall's own size brick sits on the bed, its face flush with the wall face and centred in the opening.
"""

# No `from __future__ import annotations`: the task loader imports this file without registering it in
# sys.modules, and dataclasses then cannot resolve string annotations.
import functools
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import FrameType
from typing import Any

import numpy as np
from yam_arena import demo as _demo
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP: float = TABLE_TOP[SCENE]

# --- units (m) -------------------------------------------------------------------------------------------------
# US modular brick (ASTM C216 / BIA TN 10): 7-5/8 x 3-5/8 x 2-1/4 in, laid with 3/8 in joints.
UNIT = (0.194, 0.092, 0.057)  # length, depth (bed width), height
JOINT = 0.010
COURSE = UNIT[2] + JOINT  # 67 mm
BRICK_MASS = 1.9  # kg, a cored face brick
NORMAN = (0.292, 0.092, 0.057)  # 11-5/8 x 3-5/8 x 2-1/4 in: right colour, wrong length
NORMAN_MASS = 2.8
DENSITY = 1900.0  # kg/m^3, fired clay, for the broken pieces

BRICK_RGB = (0.62, 0.25, 0.18)
BROKEN_RGB = (0.55, 0.23, 0.17)
MORTAR_RGB = (0.72, 0.70, 0.65)
TIMBER_RGB = (0.66, 0.50, 0.32)
BUCKET_RGB = (0.12, 0.12, 0.13)

# --- the wall ----------------------------------------------------------------------------------------------------
X_FACE = 0.48  # the wall face (toward the arms)
COURSES = 6
POCKET_COURSE = 3  # counted from 0 at the table
WALL_HALF = 0.301  # half the wall's length: three bricks and two joints
MORTAR_RECESS = 0.004  # joints tooled 4 mm back from the face
BED = TOP + POCKET_COURSE * COURSE - JOINT  # the pocket's floor: the top of the brick below (the bed joint is cut out)
CEILING = TOP + (POCKET_COURSE + 1) * COURSE  # the underside of the brick above
POCKET_HALF = UNIT[0] / 2 + JOINT  # 107 mm: the brick and both head joints are cut out
POCKET = (X_FACE, X_FACE + UNIT[1], -POCKET_HALF, POCKET_HALF, BED, CEILING)  # x0, x1, y0, y1, z0, z1


def _brick_row(course: int) -> list[tuple[float, float]]:
    """(centre y, length) of each brick in a course, running bond: three whole bricks in odd courses, halves at
    the ends and two whole bricks in even ones."""
    step = UNIT[0] + JOINT
    if course % 2:
        return [(-step, UNIT[0]), (0.0, UNIT[0]), (step, UNIT[0])]
    half = (UNIT[0] - JOINT) / 2
    end = WALL_HALF - half / 2
    return [(-end, half), (-step / 2, UNIT[0]), (step / 2, UNIT[0]), (end, half)]


def _wall() -> list[Box]:
    parts = []
    for course in range(COURSES):
        z = TOP + course * COURSE + UNIT[2] / 2
        for k, (y, length) in enumerate(_brick_row(course)):
            if course == POCKET_COURSE and abs(y) < 1e-9:
                continue  # the cut-out brick
            parts.append(
                Box(f"wall_c{course}_{k}", pos=(X_FACE + UNIT[1] / 2, y, z), size=(UNIT[1], length, UNIT[2]),
                    color=BRICK_RGB)
            )
    # Mortar: four slabs filling the wall's joints around the pocket, 4 mm back from the face.
    x0, x1 = X_FACE + MORTAR_RECESS, X_FACE + UNIT[1]
    top = TOP + COURSES * COURSE - JOINT

    def slab(name: str, y0: float, y1: float, z0: float, z1: float) -> Box:
        return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
                   color=MORTAR_RGB)

    parts += [
        slab("mortar_below", -WALL_HALF, WALL_HALF, TOP, BED),
        slab("mortar_above", -WALL_HALF, WALL_HALF, CEILING, top),
        slab("mortar_right", -WALL_HALF, -POCKET_HALF, BED, CEILING),
        slab("mortar_left", POCKET_HALF, WALL_HALF, BED, CEILING),
    ]
    # The inner layer (and the collar joint): the back of the pocket.
    back = 0.102
    parts.append(
        Box("wall_inner_layer", pos=(X_FACE + UNIT[1] + back / 2, 0.0, (TOP + top) / 2),
            size=(back, 2 * WALL_HALF, top - TOP), color=BRICK_RGB)
    )
    return parts


# --- broken pieces of the old brick, in the pocket ----------------------------------------------------------------
# (name, size x, y, z, centre x from the face, centre y). Each no wider than 80 mm across, so a gripper (94 mm
# open) can take it from the side.
PIECES = (
    ("broken_piece_1", (0.070, 0.062, 0.050), 0.040, -0.050),
    ("broken_piece_2", (0.060, 0.058, 0.045), 0.040, 0.050),
)


def _pieces() -> list[Box]:
    return [
        Box(name, pos=(X_FACE + dx, y, BED + size[2] / 2 + 0.001), size=size,
            mass=DENSITY * size[0] * size[1] * size[2], color=BROKEN_RGB)
        for name, size, dx, y in PIECES
    ]


# --- the bankers and the replacement bricks --------------------------------------------------------------------------
BANKER_TOP = TOP + 0.175
BANKER_X = (0.32, 0.42)  # 2 cm nearer the wall than probe 45, so the left arm reaches its grip on the brick exactly
OVERHANG = 0.04  # each brick overhangs its board's front edge, so a gripper can take its front edge
# Both bankers stand left of the pocket, so the right arm clears the pocket without passing over them.
BRICK_Y = 0.225  # the matching brick, nearer the pocket (2.5 cm clear of the hand spreading mortar at the pocket's end)
NORMAN_Y = 0.505  # the long one, farther left
BRICK_START = (BANKER_X[0] - OVERHANG + UNIT[1] / 2, BRICK_Y, BANKER_TOP + UNIT[2] / 2 + 0.001)
NORMAN_START = (BANKER_X[0] - OVERHANG + NORMAN[1] / 2, NORMAN_Y, BANKER_TOP + NORMAN[2] / 2 + 0.001)


def _bankers() -> list[Box]:
    height = BANKER_TOP - TOP
    out = []
    for name, y, length in (("banker_near", BRICK_Y, 0.24), ("banker_far", NORMAN_Y, 0.28)):
        out.append(
            Box(name, pos=((BANKER_X[0] + BANKER_X[1]) / 2, y, TOP + height / 2),
                size=(BANKER_X[1] - BANKER_X[0], length, height), color=TIMBER_RGB)
        )
    return out


REPLACEMENT = Box("replacement_brick", pos=BRICK_START, size=(UNIT[1], UNIT[0], UNIT[2]), mass=BRICK_MASS,
                  color=BRICK_RGB)
LONG_BRICK = Box("norman_brick", pos=NORMAN_START, size=(NORMAN[1], NORMAN[0], NORMAN[2]), mass=NORMAN_MASS,
                 color=BRICK_RGB)

# --- the mortar station: a board with fresh mortar on its far end, and a small trowel ------------------------------
# On the right, just past the end of the wall: a timber board 17 cm high runs toward the wall, the far part of its
# top a patch of fresh mortar (flush with the timber). A small rectangular trowel (a buttering trowel)
# lies on its near end along x, blade toward the wall, its handle propped level on a small block and sticking out
# over the board's front edge so a hand can take its end.
MORTAR_RGB_FRESH = (0.45, 0.44, 0.42)  # wet mortar, darker than the cured joints
STEEL_RGB = (0.62, 0.64, 0.66)
BOARD_TOP = TOP + 0.17
BOARD_Y = (-0.385, -0.305)  # clear of the wall's end (y -0.301)
BOARD_X = (0.295, 0.70)
PATCH_X = (0.53, 0.66)  # the mortar, across the board's width
# The trowel: a wooden handle (the rigid body, as a Box) with a steel shank and blade added by the hook below.
# Trowel frame: origin at the handle's centre, x along the trowel toward the blade, z up.
HANDLE_SIZE = (0.110, 0.032, 0.032)  # a 32 mm square wooden handle (25 mm let the trowel tip in the grip)
TROWEL_RISE = 0.025  # the offset shank: the handle sits this much above the blade
BLADE_SIZE = (0.127, 0.070, 0.002)  # a 5 x 2-3/4 in blade (a buttering trowel): wide enough that one pass
# covers the pocket's ends with the hand staying clear of the replacement brick
BLADE_X = (HANDLE_SIZE[0] / 2 + 0.005, HANDLE_SIZE[0] / 2 + 0.005 + BLADE_SIZE[0])  # in the trowel frame
BLADE_TOP = -(HANDLE_SIZE[2] / 2 + TROWEL_RISE)  # in the trowel frame: 37.5 mm below the handle's centre
BLADE_UNDER = BLADE_TOP - BLADE_SIZE[2]  # 39.5 mm below it
TROWEL_MASS = 0.25
TROWEL_START = (0.31, sum(BOARD_Y) / 2, BOARD_TOP - BLADE_UNDER + 0.001)
TROWEL_PARTS = (  # (name, centre, size) in the trowel frame; all one rigid body with the handle
    ("blade", ((BLADE_X[0] + BLADE_X[1]) / 2, 0.0, BLADE_UNDER + BLADE_SIZE[2] / 2), BLADE_SIZE),
    ("shank", (HANDLE_SIZE[0] / 2 + 0.0025, 0.0, (BLADE_TOP - HANDLE_SIZE[2] / 2) / 2),
     (0.005, 0.012, -HANDLE_SIZE[2] / 2 - BLADE_TOP)),
)
TROWEL = Box("trowel", pos=TROWEL_START, size=HANDLE_SIZE, mass=TROWEL_MASS, color=TIMBER_RGB)
_REST_X = (TROWEL_START[0] + 0.005, TROWEL_START[0] + 0.04)  # under the handle, clear of the end a hand takes


def _mortar_station() -> list[Box]:
    height = BOARD_TOP - TOP
    y0, y1 = BOARD_Y

    def slab(name: str, x0: float, x1: float, rgb: tuple[float, float, float]) -> Box:
        return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, TOP + height / 2), size=(x1 - x0, y1 - y0, height),
                   color=rgb)

    rest_h = TROWEL_START[2] - HANDLE_SIZE[2] / 2 - BOARD_TOP
    return [
        slab("mortar_board_near", BOARD_X[0], PATCH_X[0], TIMBER_RGB),
        slab("fresh_mortar", PATCH_X[0], PATCH_X[1], MORTAR_RGB_FRESH),
        slab("mortar_board_far", PATCH_X[1], BOARD_X[1], TIMBER_RGB),
        Box("trowel_rest", pos=((_REST_X[0] + _REST_X[1]) / 2, TROWEL_START[1], BOARD_TOP + rest_h / 2),
            size=(_REST_X[1] - _REST_X[0], 0.03, rest_h), color=TIMBER_RGB),
    ]


# --- the debris bucket ----------------------------------------------------------------------------------------------
BUCKET_CENTER = (0.20, -0.48)  # behind the right arm, clear of the mortar board
BUCKET_INNER = 0.17
BUCKET_HEIGHT = 0.13
BUCKET_WALL = 0.008


def _bucket() -> list[Box]:
    cx, cy = BUCKET_CENTER
    w, h, t = BUCKET_INNER, BUCKET_HEIGHT, BUCKET_WALL
    outer = w + 2 * t
    z = TOP + h / 2
    return [
        Box("bucket_floor", pos=(cx, cy, TOP + t / 2), size=(outer, outer, t), color=BUCKET_RGB),
        Box("bucket_near", pos=(cx - (w + t) / 2, cy, z), size=(t, outer, h), color=BUCKET_RGB),
        Box("bucket_far", pos=(cx + (w + t) / 2, cy, z), size=(t, outer, h), color=BUCKET_RGB),
        Box("bucket_right", pos=(cx, cy - (w + t) / 2, z), size=(w, t, h), color=BUCKET_RGB),
        Box("bucket_left", pos=(cx, cy + (w + t) / 2, z), size=(w, t, h), color=BUCKET_RGB),
    ]


OBJECTS = [*_wall(), *_pieces(), *_bankers(), REPLACEMENT, LONG_BRICK, *_bucket(), *_mortar_station(), TROWEL]

# --- scoring -------------------------------------------------------------------------------------------------------
FLUSH_TOL = 0.003  # face within 3 mm of the wall face (BIA: joint tolerance +-1/8 in)
BED_TOL = 0.003  # bottom within 3 mm of the pocket's bed
CENTRE_TOL = 0.005  # head joints 5 to 15 mm
TURN_TOL = math.radians(3.0)  # long side within 3 degrees of the wall line, bed within 3 degrees of level
PIECE_POINTS = 0.15  # each broken piece in the bucket
MORTAR_POINTS = 0.3  # the pocket's bed covered with fresh mortar before the brick goes in (see MORTAR_STEPS)
# (share of bed cells, share of MORTAR_POINTS): a bed counts as full at 98% (at most 3 of its 189 cells, about 3 cm2,
# bare); a bed joint is full in practice well before every square centimetre is covered.
MORTAR_STEPS = ((0.98, 1.0), (0.95, 0.8), (0.85, 0.5), (0.60, 0.25))
SEATED_POINTS = 0.4  # the matching brick seated flush
IN_POCKET_POINTS = 0.2  # ... or resting in the pocket, at least half its depth in, but not flush
HOLD_STEPS = 20  # 1 s at 20 Hz: a full score counts once it has held this long
HELD_BACK = 0.9  # what score() reports for a full score that has not yet held
SCORE_OVER = "end"


def rotation(quat_xyzw: Any) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def in_bucket(pos: np.ndarray) -> bool:
    cx, cy = BUCKET_CENTER
    half = BUCKET_INNER / 2
    return bool(abs(pos[0] - cx) <= half and abs(pos[1] - cy) <= half and TOP < pos[2] < TOP + BUCKET_HEIGHT)


@dataclass
class BrickFit:
    face_gap: float  # m the brick's face sits behind (+) or proud of (-) the wall face
    depth_in: float  # m of its 92 mm depth inside the pocket
    bed_gap: float  # m its bottom is above the pocket's bed
    off_centre: float  # m from the opening's centre, along the wall
    turn: float  # rad its long side is turned from the wall line
    tilt: float  # rad its bed is tilted from level

    @property
    def seated(self) -> bool:
        return (
            abs(self.face_gap) <= FLUSH_TOL
            and abs(self.bed_gap) <= BED_TOL
            and abs(self.off_centre) <= CENTRE_TOL
            and self.turn <= TURN_TOL
            and self.tilt <= TURN_TOL
        )

    @property
    def in_pocket(self) -> bool:
        """Resting on the pocket's bed, at least half its depth inside, roughly square to the wall."""
        return (
            self.depth_in >= UNIT[1] / 2
            and abs(self.bed_gap) <= BED_TOL
            and abs(self.off_centre) <= JOINT
            and self.turn <= math.radians(10.0)
        )


def brick_fit(pos: np.ndarray, quat_xyzw: Any) -> BrickFit:
    rot = rotation(quat_xyzw)
    long_axis = rot[:, 1]  # the brick's length lies along its own y
    turn = math.acos(min(1.0, abs(float(long_axis[1])) / max(1e-9, float(np.linalg.norm(long_axis[:2])))))
    tilt = math.acos(min(1.0, abs(float(rot[2, 2]))))
    face = pos[0] - UNIT[1] / 2
    return BrickFit(
        face_gap=float(face - X_FACE),
        depth_in=float(min(UNIT[1], max(0.0, pos[0] + UNIT[1] / 2 - X_FACE))),
        bed_gap=float(pos[2] - UNIT[2] / 2 - BED),
        off_centre=float(pos[1]),
        turn=turn,
        tilt=tilt,
    )


# --- fresh mortar on the pocket's bed --------------------------------------------------------------------------------
# Mortar is not a paste here: it is a ledger of the bed's cells (1 cm-ish, 9 deep x 21 along the pocket), updated from
# the trowel's pose each control step and drawn as thin tiles the cameras see, the way the accepted painting task
# treats paint. The trowel starts with a load of mortar on its blade (it can take another by being laid flat on
# the board's mortar patch); a load covers LOAD_CELLS cells. A bed cell gets mortar when it lies under the loaded
# blade (its centre within BLADE_MARGIN of the blade's outline: mortar squeezes a little past the edge) with the
# blade's underside between BLADE_BELOW below and BLADE_ABOVE above the bed there, and the blade within
# MAX_BLADE_TILT of level. BLADE_ABOVE is the bed joint's thickness: a mason spreads a bed with the blade about that
# far above the brick, the mortar filling the gap. Cells covered once the replacement brick has started into the
# pocket do not count: the bed has to be mortared before the brick goes in.
BED_CELLS = (9, 21)  # along x (the pocket's depth), along y (its width)
LOAD_CELLS = 240  # one trowel load covers the 189-cell bed once with some to spare
BLADE_MARGIN = 0.003  # mortar squeezes a little past the blade's edge: cells within 3 mm of its outline count
BLADE_ABOVE = JOINT  # 10 mm: the bed joint's thickness
BLADE_BELOW = 0.003
MAX_BLADE_TILT = math.radians(10.0)
BRICK_IN = 0.005  # m of the brick inside the pocket that counts as it having started in
_gx = X_FACE + (np.arange(BED_CELLS[0]) + 0.5) * UNIT[1] / BED_CELLS[0]
_gy = -POCKET_HALF + (np.arange(BED_CELLS[1]) + 0.5) * 2 * POCKET_HALF / BED_CELLS[1]
BED_CELL_XY = np.stack(np.meshgrid(_gx, _gy, indexing="ij"), axis=-1).reshape(-1, 2)
N_BED = len(BED_CELL_XY)


# The simulator reports an object's position as its bounding box's centre; for the trowel that lies forward of and
# below the handle's centre (the trowel frame's origin) because of the blade.
TROWEL_BOX_CENTRE = np.array([(BLADE_X[1] - HANDLE_SIZE[0] / 2) / 2, 0.0, (BLADE_UNDER + HANDLE_SIZE[2] / 2) / 2])


def trowel_origin(reported: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """The trowel frame's origin (the handle's centre), from the reported (bounding-box centre) position."""
    origin: np.ndarray = reported - rot @ TROWEL_BOX_CENTRE
    return origin


def blade_cells(pos: np.ndarray, rot: np.ndarray, cells_xy: np.ndarray, surface_z: float) -> np.ndarray:
    """Which cells (on a level surface at ``surface_z``) the blade is touching, from the trowel's pose."""
    if math.acos(min(1.0, abs(float(rot[2, 2])))) > MAX_BLADE_TILT:
        return np.zeros(len(cells_xy), dtype=bool)
    world = np.column_stack([cells_xy, np.full(len(cells_xy), surface_z)])
    local = (world - pos) @ rot  # each cell in the trowel frame
    inside = ((local[:, 0] >= BLADE_X[0] - BLADE_MARGIN) & (local[:, 0] <= BLADE_X[1] + BLADE_MARGIN)
              & (np.abs(local[:, 1]) <= BLADE_SIZE[1] / 2 + BLADE_MARGIN))
    gap = BLADE_UNDER - local[:, 2]  # the blade's underside above the surface at each cell
    touching: np.ndarray = inside & (gap <= BLADE_ABOVE) & (gap >= -BLADE_BELOW)
    return touching


_PATCH_XY = np.array([[x, y] for x in np.linspace(PATCH_X[0] + 0.01, PATCH_X[1] - 0.01, 6)
                      for y in np.linspace(BOARD_Y[0] + 0.01, BOARD_Y[1] - 0.01, 4)])


@dataclass
class MortarLedger:
    bed: np.ndarray = field(default_factory=lambda: np.zeros(N_BED, dtype=bool))
    load: int = LOAD_CELLS  # cells the trowel can still cover: it starts with a load of mortar on its blade
    frozen: bool = False  # the brick has started into the pocket: no more mortar counts
    steps: int = 0
    full_steps: int = 0

    @property
    def coverage(self) -> float:
        return float(self.bed.sum()) / N_BED

    def apply(self, trowel_pos: np.ndarray, trowel_rot: np.ndarray, brick_in: bool) -> None:
        self.steps += 1
        if brick_in:
            self.frozen = True
        if blade_cells(trowel_pos, trowel_rot, _PATCH_XY, BOARD_TOP).sum() >= len(_PATCH_XY) // 4:
            self.load = LOAD_CELLS  # the blade dragged flat through the mortar
        if self.frozen or self.load <= 0:
            return
        new = blade_cells(trowel_pos, trowel_rot, BED_CELL_XY, BED) & ~self.bed
        if new.any():
            take = np.flatnonzero(new)[: self.load]
            self.bed[take] = True
            self.load -= len(take)


def mortar_share(coverage: float) -> float:
    for threshold, share in MORTAR_STEPS:
        if coverage >= threshold - 1e-9:
            return share
    return 0.0


@dataclass
class Breakdown:
    pieces_in_bucket: int
    mortar_coverage: float
    fit: BrickFit
    brick_points: float
    score: float  # before the hold


def score_state(objects: Mapping[str, Any], ledger: "MortarLedger | None" = None) -> Breakdown:
    ledger = _LEDGER if ledger is None else ledger
    pieces = sum(in_bucket(np.asarray(objects[name].pos, dtype=float)) for name, *_ in PIECES)
    brick = objects[REPLACEMENT.name]
    fit = brick_fit(np.asarray(brick.pos, dtype=float), brick.quat_xyzw)
    points = SEATED_POINTS if fit.seated else IN_POCKET_POINTS if fit.in_pocket else 0.0
    mortar = MORTAR_POINTS * mortar_share(ledger.coverage)
    return Breakdown(pieces, ledger.coverage, fit, points, round(PIECE_POINTS * pieces + mortar + points, 6))


_LEDGER = MortarLedger()
_last_objects: object = None  # the step's state dict, kept alive so its id can't be reused
_hooked = False
_hooked_to: Any = None  # the embodiment whose reset is wrapped
_warned: set[str] = set()
RESET_TAG = "replace_brick_new_episode"  # on a wrapped reset: the new_episode it calls


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(f"replace_brick: {message}", RuntimeWarning, stacklevel=3)


def new_episode() -> None:
    """Start an empty ledger and clear the fresh-mortar tiles."""
    global _LEDGER, _last_objects
    _LEDGER = MortarLedger()
    _last_objects = None
    _TILES.hide()


def _calling_embodiment() -> Any:
    """The embodiment whose step is calling us, found by what it has (an episode clock, a task, a reset)."""
    frame: FrameType | None = sys._getframe(2)
    while frame is not None:
        candidate = frame.f_locals.get("self")
        if (
            candidate is not None
            and hasattr(candidate, "_t")
            and hasattr(candidate, "task")
            and callable(getattr(candidate, "reset", None))
        ):
            return candidate
        frame = frame.f_back
    return None


def _hook_resets() -> None:
    """Task files get no reset hook, so wrap the calling embodiment's reset: every new trial then starts a fresh
    ledger and hides the last trial's tiles. If no embodiment is found, the trowel resting at its start pose starts a
    new trial instead."""
    global _hooked, _hooked_to
    embodiment = _calling_embodiment()
    if embodiment is None:
        if not _hooked:
            _warn_once("reset", "could not find the embodiment's reset; a trowel resting at its start starts a new trial")
        return
    if embodiment is _hooked_to:
        return
    original = embodiment.reset
    if getattr(original, RESET_TAG, None) is new_episode:  # already wrapped by this file
        _hooked, _hooked_to = True, embodiment
        return

    @functools.wraps(original)
    def reset(*args: Any, **kwargs: Any) -> Any:
        new_episode()
        return original(*args, **kwargs)

    setattr(reset, RESET_TAG, new_episode)
    try:
        embodiment.reset = reset
        _hooked, _hooked_to = True, embodiment
    except (AttributeError, TypeError):
        _warn_once("reset", "could not hook the embodiment's reset; a trowel resting at its start starts a new trial")


def _update(objects: Mapping[str, Any]) -> MortarLedger:
    """Advance the mortar ledger and the full-score count once per control step (success() and score() share the
    step's state dict)."""
    global _last_objects
    if objects is _last_objects:
        return _LEDGER
    try:
        _hook_resets()
    except Exception as error:  # an odd caller on the stack: carry on without the hook
        _warn_once("hook", f"could not look for the embodiment ({type(error).__name__}: {error})")
    trowel = objects[TROWEL.name]
    rot = rotation(trowel.quat_xyzw)
    pos = trowel_origin(np.asarray(trowel.pos, dtype=float), rot)
    at_start = np.linalg.norm(pos - np.asarray(TROWEL_START)) < 0.003
    resting = np.linalg.norm(np.asarray(trowel.lin_vel, dtype=float)) < 0.01
    if not _hooked and at_start and resting and _LEDGER.steps > 0 and _LEDGER.coverage == 0.0:
        new_episode()
    _last_objects = objects
    brick = objects[REPLACEMENT.name]
    brick_in = brick_fit(np.asarray(brick.pos, dtype=float), brick.quat_xyzw).depth_in > BRICK_IN
    _LEDGER.apply(pos, rot, brick_in)
    full = score_state(objects, _LEDGER).score >= 1.0 - 1e-9
    _LEDGER.full_steps = _LEDGER.full_steps + 1 if full else 0
    _TILES.draw(objects, _LEDGER)
    return _LEDGER


def _held(ledger: MortarLedger, raw: float) -> float:
    """A full score counts once it has held for HOLD_STEPS control steps (1 s); until then it reports HELD_BACK."""
    return raw if raw < 1.0 - 1e-9 or ledger.full_steps >= HOLD_STEPS else HELD_BACK


# --- fresh mortar the cameras can see ---------------------------------------------------------------------------------


class _Tiles:
    """Mortared bed cells drawn as thin grey tiles, and a layer of mortar on the trowel's blade while it is loaded,
    with Isaac Lab's VisualizationMarkers (imported lazily: the simulator is running by the time score() is first
    called). Only a picture of the ledger: if drawing fails for any reason, it turns off with one warning and scoring
    goes on. Outside the simulator (unit tests), nothing is drawn."""

    def __init__(self) -> None:
        self._bed: Any = None
        self._blade: Any = None
        self._origin = np.zeros(3)
        self._shown = (-1, False)
        self._off = False

    def hide(self) -> None:
        if self._bed is None or self._off:
            return
        try:
            self._bed.set_visibility(False)
            self._blade.set_visibility(False)
            self._shown = (0, False)
        except Exception as error:  # cosmetic only: never abort a trial over the picture
            self._disable(error)

    def draw(self, objects: Mapping[str, Any], ledger: MortarLedger) -> None:
        if self._off:
            return
        count, loaded = int(ledger.bed.sum()), ledger.load > 0
        try:
            if self._bed is None:
                self._origin = self._env_origin(objects)
                self._bed, self._blade = self._make()
            if count != self._shown[0]:
                self._bed.set_visibility(count > 0)
                if count:
                    xy = BED_CELL_XY[ledger.bed]
                    points = np.column_stack([xy, np.full(count, BED + 0.0007)]) + self._origin
                    self._bed.visualize(translations=points.astype(np.float32),
                                        orientations=np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)).astype(np.float32))
            self._blade.set_visibility(loaded)
            if loaded:
                trowel = objects[TROWEL.name]
                rot = rotation(trowel.quat_xyzw)
                centre = trowel_origin(np.asarray(trowel.pos, dtype=float), rot) + rot @ np.array(
                    [(BLADE_X[0] + BLADE_X[1]) / 2 + 0.005, 0.0, BLADE_TOP + 0.003])
                self._blade.visualize(translations=(centre + self._origin)[None].astype(np.float32),
                                      orientations=np.asarray(trowel.quat_xyzw, dtype=np.float32)[None])
            self._shown = (count, loaded)
        except ImportError:
            self._off = True  # no simulator: unit tests
        except Exception as error:  # cosmetic only: never abort a trial over the picture
            self._disable(error)

    def _disable(self, error: Exception) -> None:
        self._off = True
        _warn_once("tiles", f"mortar tiles turned off ({type(error).__name__}: {error}); scoring is unaffected")

    @staticmethod
    def _make() -> tuple[Any, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        look = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(_linear(c) for c in MORTAR_RGB_FRESH), roughness=0.9)
        dx, dy = UNIT[1] / BED_CELLS[0], 2 * POCKET_HALF / BED_CELLS[1]
        bed = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/fresh_mortar_bed",
            markers={"mortar": sim_utils.CuboidCfg(size=(dx * 1.02, dy * 1.02, 0.0012), visual_material=look)}))
        blade = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/fresh_mortar_on_trowel",
            markers={"mortar": sim_utils.CuboidCfg(size=(BLADE_SIZE[0] * 0.8, BLADE_SIZE[1] * 0.8, 0.006),
                                                   visual_material=look)}))
        return bed, blade

    @staticmethod
    def _env_origin(objects: Mapping[str, Any]) -> np.ndarray:
        """World offset of the env (object states are env-relative; markers live in world coordinates)."""
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            if prim.GetName() == "wall_inner_layer":
                world = np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=float)
                offset: np.ndarray = world - np.asarray(objects["wall_inner_layer"].pos, dtype=float)
                return offset
        return np.zeros(3)


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


_TILES = _Tiles()


def score(objects: Mapping[str, Any]) -> float:
    """0.15 per broken piece in the bucket; up to 0.3 for the pocket's bed mortared before the brick goes in (see
    MORTAR_STEPS); 0.4 for the matching brick seated flush (0.2 if it rests in the pocket but is not flush). A full
    score counts once it has held for 1 s."""
    ledger = _update(objects)
    raw = score_state(objects, ledger).score
    return _held(ledger, raw)


def success(objects: Mapping[str, Any]) -> bool:
    """Full score. The embodiment ends the trial once this has held for 1 s, the same hold score() applies."""
    ledger = _update(objects)
    return score_state(objects, ledger).score >= 1.0 - 1e-9


# --- the scripted demo ----------------------------------------------------------------------------------------------
# yam_arena's scripted player only grips straight down (Waypoint.yaw turns that grip about the vertical). Here a
# brick goes into a pocket under the course above, so the demo grips from the side. A waypoint's yaw may be a
# SideGrip, a float that also carries the side grip's orientation; the hook below makes yam_arena.demo's
# grip_rotation honour it and leaves every other waypoint as it was.


class SideGrip(float):
    """The gripper pointing at the wall (+x), pitched ``pitch`` rad down and turned ``yaw`` rad toward +y about the
    vertical. ``fingers="z"``: the fingers close vertically (above and below a brick); ``"y"``: they close along the
    wall (either side of a piece)."""

    def __new__(cls, fingers: str = "z", pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> "SideGrip":
        out = super().__new__(cls, 0.0)
        out.fingers, out.pitch, out.yaw, out.roll = fingers, pitch, yaw, roll
        return out

    def matrix(self, down: np.ndarray) -> np.ndarray:
        def about(axis: int, angle: float) -> np.ndarray:
            c, s = math.cos(angle), math.sin(angle)
            i, j = [(1, 2), (2, 0), (0, 1)][axis]
            m = np.eye(3)
            m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
            return m

        level = about(1, -math.pi / 2) @ down  # approach turned from straight down to +x; fingers still along y
        if self.fingers == "z":
            level = about(0, SIDE_ROLL) @ level
        return about(2, self.yaw) @ about(1, self.pitch) @ about(0, self.roll) @ level


SIDE_ROLL = math.pi / 2  # which way the hand turns to close its fingers vertically (joint limits decide)


def _install_side_grips() -> None:
    original = _demo.grip_rotation
    if getattr(original, "side_grips", False):
        return

    def grip_rotation(yaw: float = 0.0) -> np.ndarray:
        if isinstance(yaw, SideGrip):
            return yaw.matrix(original(0.0))
        return original(yaw)

    grip_rotation.side_grips = True  # type: ignore[attr-defined]
    _demo.grip_rotation = grip_rotation


_install_side_grips()

BRICK_GRIP = SideGrip("z")
PIECE_GRIP = SideGrip("y")
PAD_DEPTH = 0.035  # how far the pads reach over the brick's front edge (20 mm let it pivot in the grip, probe 6)
TROWEL_SAG = 0.009  # the trowel (0.25 kg) held out on one arm: commanded this much higher (probe 29: about 9 mm)
TROWEL_PRESS = -0.002  # the blade's underside commanded this far below a surface, i.e. 2 mm above it: inside the
# ledger's contact band, but not pushing on the blade, which tipped the trowel in the grip (probes 30-31)
TROWEL_GRIP_DEPTH = 0.045  # how far along the handle from its end the pads reach
TROWEL_LIFT = 0.07  # lifted this far straight up off its rest (commanded) before it moves anywhere
TROWEL_DROOP = math.radians(2.0)  # held by its handle's sides, the blade droops this much (probes 37-39: 2-4 deg)
TROWEL_ROLL = math.radians(6.0)  # held, the blade also leans about 6 deg across its width (its +y edge low, probes
# 41-45), dragging that edge on the bed; the hand turns this much about the handle to level it
BLADE_CLEAR = 0.010  # the blade's height above the bed while it slides in or out of the pocket
BED_PRESS = -0.005  # over the pocket's bed the blade rides 5 mm above it, inside the 10 mm joint: resting on the
# bed, its edge caught as it moved and twisted the trowel in the grip (probe 40)
SPREAD_Y = (-0.07, 0.07)  # the blade's centre runs along the pocket between these (y): its 70 mm reach both ends
SPREAD_YAW = 0.0  # the trowel held straight (turning it about the vertical twisted it out of the grip, probe 29)
GRIP_SPREAD = 0.065  # each arm holds the brick this far from its centre (32 mm in from its ends)
LIFT = 0.010  # the brick's bottom is carried this far above the pocket's bed
ENTRY_RISE = 0.022  # the hands rise this much while pushing the brick into the pocket (see the demo); at 10 mm
# the brick reached the wall face only level with the bed and its lowered nose caught (probe 48)
# Measured (probes 5-7): carrying the 1.9 kg brick between them, the arms (position-controlled, shoulder stiffness
# 80 N m/rad) hold it below where they are commanded, more the farther they reach: about 44 mm with the TCP at
# x 0.28 m and 65 mm at x 0.39 m. Every waypoint with the brick in hand is raised by sag(x).
SAG_AT = (0.28, 0.044)
SAG_PER_M = 0.19


SAG_MAX = 0.065  # the most measured; not extrapolated past it


def sag(tcp_x: float) -> float:
    return min(SAG_MAX, SAG_AT[1] + SAG_PER_M * (tcp_x - SAG_AT[0]))


# ... and the brick hangs about 4-5 degrees back-end down in the pinch, so the hands carry it tilted up this much.
CARRY_PITCH = -math.radians(5.0)
INSERT_DEPTH = 0.056  # set down with this much of its 92 mm depth in the pocket: past its centre, front 36 mm out
DROP_AWAY = 0.012  # after letting go, the hands drop this far so the lower finger leaves the brick before backing off
# Fingertips commanded this far past flush: the arms give way under the ~22 N it takes to slide the brick, and the
# inner layer stops the brick at flush, so the extra only presses it home and squares it to the wall.
PUSH_PAST = 0.04  # probe 12: 20 mm left the brick 9 mm short, turned 5.5 degrees
BACK_OFF = 0.08  # after letting go, the hands back off clear of the brick's edge before closing their fingers
# Easing the brick in: each step loosens the grip, slides the hands 10 mm back along the brick, grips again and
# pushes it 10 mm deeper.
EASE_STEPS = 3
EASE_STEP = 0.010
EASE_PUSH_PAST = 0.004  # commanded past each 10 mm push: the arms give a little under the ~22 N of sliding friction
LOOSE = -(UNIT[2] + 2 * 0.006) / 2  # finger target leaving 6 mm of play above and below the brick (69 mm open)


class Plan:
    """Waypoints for both arms in step: every move lasts as long on each arm."""

    def __init__(self) -> None:
        self.arms: dict[str, list[Waypoint]] = {"left": [], "right": []}

    def move(self, duration: float, label: str, **targets: tuple[tuple[float, float, float], float, float]) -> None:
        for arm, plan in self.arms.items():
            if arm in targets:
                pos, finger, yaw = targets[arm]
                plan.append(Waypoint(duration, pos, finger, label, yaw))
            elif plan:
                last = plan[-1]
                plan.append(Waypoint(duration, last.pos, last.finger, "wait", last.yaw))
            else:
                raise ValueError(f"{arm} has no pose yet; give it one in the first move")


def tcp_at_front(front_x: float, y: float, z: float, depth: float = PAD_DEPTH) -> tuple[float, float, float]:
    """The TCP whose pad tips reach ``depth`` past ``front_x``."""
    return (front_x + depth - PAD_TIP_PAST_TCP, y, z)


def demo() -> dict[str, list[Waypoint]]:
    """Right arm: take the two broken pieces out of the pocket into the bucket, then spread fresh mortar on the
    pocket's bed with the trowel. Then both arms take the replacement brick by its front edge, set it on the
    pocket's bed partway in, ease it in, and push it flush with the wall."""
    plan = Plan()
    park = {"left": (0.25, BRICK_Y + GRIP_SPREAD, TOP + 0.30), "right": (0.25, -0.20, TOP + 0.30)}
    plan.move(1.5, "start", left=(park["left"], FINGER_OPEN, BRICK_GRIP), right=(park["right"], FINGER_OPEN, PIECE_GRIP))

    # 1. The broken pieces, front one first, by the right arm.
    # The arm travels high (above the bankers and the brick on them) and only comes down in front of the pocket,
    # where nothing stands on the table.
    high = TOP + 0.30
    bucket_drop = (BUCKET_CENTER[0], BUCKET_CENTER[1], TOP + BUCKET_HEIGHT + 0.06)
    for name, size, dx, y in PIECES:
        z = BED + size[2] / 2 + 0.002
        front = X_FACE + dx - size[0] / 2
        grip = tcp_at_front(front, y, z, depth=size[0] / 2 + 0.01)
        out = (X_FACE - 0.10, y, z + 0.004)
        plan.move(2.0, "above the pocket", right=((X_FACE - 0.10, y, high), FINGER_OPEN, PIECE_GRIP))
        plan.move(1.5, f"in front of {name}", right=((X_FACE - 0.08, y, z), FINGER_OPEN, PIECE_GRIP))
        plan.move(1.5, f"reach {name}", right=(grip, FINGER_OPEN, PIECE_GRIP))
        plan.move(0.8, "grip", right=(grip, FINGER_CLOSED, PIECE_GRIP))
        plan.move(0.5, "lift", right=((grip[0], y, z + 0.004), FINGER_CLOSED, PIECE_GRIP))
        plan.move(1.5, "draw it out", right=(out, FINGER_CLOSED, PIECE_GRIP))
        plan.move(1.0, "up", right=((out[0], y, high), FINGER_CLOSED, PIECE_GRIP))
        # Over to the bucket at travel height and straight down: a slanting line runs over the trowel's handle.
        plan.move(2.0, "above the bucket", right=((bucket_drop[0], bucket_drop[1], high), FINGER_CLOSED, PIECE_GRIP))
        plan.move(1.0, "down to the bucket", right=(bucket_drop, FINGER_CLOSED, PIECE_GRIP))
        plan.move(0.8, "drop", right=(bucket_drop, FINGER_OPEN, PIECE_GRIP))
        plan.move(1.0, "clear of the bucket", right=((bucket_drop[0], bucket_drop[1], high), FINGER_OPEN, PIECE_GRIP))

    # 2. Fresh mortar on the pocket's bed, by the right arm: take the trowel by its handle's end (fingers either side
    # below), drag its blade flat through the mortar on the board, carry it to the pocket, slide the blade in flat
    # on the bed, spread along the pocket in one pass (the blade is longer than the pocket is deep), and put the
    # trowel back.
    ty = TROWEL_START[1]
    to_blade = (BLADE_X[0] + BLADE_X[1]) / 2 + HANDLE_SIZE[0] / 2  # from the handle's end to the blade's centre
    blade0 = TROWEL_START[0] - HANDLE_SIZE[0] / 2 + to_blade  # the blade's centre (x) at rest

    def trowel(duration: float, label: str, blade_x: float, blade_y: float, underside: float, finger: float,
               held: bool = True, yaw: float = 0.0) -> None:
        """The right hand on the handle's end, fingers either side of it (as for the broken pieces: the pads along
        the handle hold it against twisting), with the blade's centre at (blade_x, blade_y), its underside at
        ``underside``, and the trowel turned ``yaw`` rad toward +y about the vertical (kept at 0: turning it twisted
        it out of the grip, probe 29). Nothing pushes on the blade and every move is slow: in this grip a push on the
        blade, or a sharp start or stop, tips it (probes 30-34)."""
        z = underside - BLADE_UNDER + (TROWEL_SAG if held else 0.0)
        end = np.array([blade_x, blade_y]) - to_blade * np.array([math.cos(yaw), math.sin(yaw)])
        tcp = end + (TROWEL_GRIP_DEPTH - PAD_TIP_PAST_TCP) * np.array([math.cos(yaw), math.sin(yaw)])
        pitch = -TROWEL_DROOP if held else 0.0  # held, the hand tilts up as much as the blade droops in the grip
        plan.move(duration, label, right=((float(tcp[0]), float(tcp[1]), z), finger,
                                          SideGrip("y", pitch=pitch, yaw=yaw, roll=TROWEL_ROLL if held else 0.0)))

    at_rest = TROWEL_START[2] - 0.001 + BLADE_UNDER
    trowel(2.0, "above the trowel", blade0 - 0.05, ty, high - 0.05, FINGER_OPEN, held=False)
    trowel(1.5, "in front of the handle", blade0 - 0.05, ty, at_rest, FINGER_OPEN, held=False)
    trowel(1.0, "reach the handle's end", blade0, ty, at_rest, FINGER_OPEN, held=False)
    trowel(0.8, "grip", blade0, ty, at_rest, FINGER_CLOSED, held=False)
    # Straight up well clear first: with 3 cm commanded it rose only about 1 cm, the blade drooped a few degrees,
    # and its tip still dragged on the board as the hand pulled back, so the fingers slid off the handle's end
    # (probes 34-36).
    trowel(2.0, "lift it straight up", blade0, ty, at_rest + TROWEL_LIFT, FINGER_CLOSED)
    # The trowel already carries mortar (as a mason picks it up). Every move with it held is slow: a sharp start or
    # stop tips the blade in the grip (probe 32).
    reach = BLADE_SIZE[0] / 2 * math.cos(SPREAD_YAW) + BLADE_SIZE[1] / 2 * math.sin(SPREAD_YAW)
    in_x = X_FACE + UNIT[1] - 0.006 - reach  # 6 mm off the back: at 2 mm its corner jammed on a small twist (probe 39)
    out_x = X_FACE - 0.04 - reach  # its tip 4 cm in front of the wall face: at 1 cm it caught the wall's end (probe 33)
    trowel(2.5, "draw it back", out_x, ty, at_rest + TROWEL_LIFT, FINGER_CLOSED)
    # Across at 2 cm above the bed (clear of the board, whose edge the blade caught in probe 35), then down in
    # front of the pocket.
    trowel(3.0, "over to the pocket", out_x, 0.0, BED + 0.02, FINGER_CLOSED, yaw=SPREAD_YAW)
    # In and out through the middle of the opening, 7 cm clear of its sides (at the pocket's end the blade's corner
    # caught the wall beside it, probe 38), and along the trowel only clear of the bed (a blade dragged along its
    # length tips in the grip, probe 37). On the bed it moves only crosswise: to one end, to the other, back to the
    # middle (drawn out at the pocket's end, the blade's edge would sweep the replacement brick's end, probe 32).
    trowel(1.0, "down in front of it", out_x, 0.0, BED + BLADE_CLEAR, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(2.0, "slide the blade in", in_x, 0.0, BED + BLADE_CLEAR, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(1.0, "down to the bed", in_x, 0.0, BED - BED_PRESS, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(2.0, "spread to one end", in_x, SPREAD_Y[0], BED - BED_PRESS, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(3.5, "spread to the other end", in_x, SPREAD_Y[1], BED - BED_PRESS, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(2.0, "back to the middle", in_x, 0.0, BED - BED_PRESS, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(0.8, "lift off the bed", in_x, 0.0, BED + BLADE_CLEAR, FINGER_CLOSED, yaw=SPREAD_YAW)
    trowel(1.5, "out of the pocket", out_x, 0.0, BED + BLADE_CLEAR, FINGER_CLOSED, yaw=SPREAD_YAW)
    # Back along the front, the blade's tip 4 cm in front of the wall, to in front of the board, then forward over
    # it: the straight line from the pocket runs the tip into the wall's end (probe 40).
    trowel(3.0, "back along the wall", out_x, ty, at_rest + TROWEL_LIFT, FINGER_CLOSED)
    trowel(1.5, "over the board", blade0, ty, at_rest + TROWEL_LIFT, FINGER_CLOSED)
    trowel(1.5, "lower it", blade0, ty, at_rest + 0.015, FINGER_CLOSED)
    trowel(1.5, "set it down", blade0, ty, at_rest - 0.003, FINGER_CLOSED)
    trowel(0.6, "let go", blade0, ty, at_rest, FINGER_OPEN, held=False)
    trowel(1.0, "back away", blade0 - 0.06, ty, at_rest + 0.03, FINGER_OPEN, held=False)

    # 3. The brick: both arms, fingers above and below its front edge, 32 mm in from each end.
    zb = BRICK_START[2] - 0.001
    front0 = BRICK_START[0] - UNIT[1] / 2
    ys = {"left": BRICK_Y + GRIP_SPREAD, "right": BRICK_Y - GRIP_SPREAD}
    park_right = (park["right"][0], BRICK_Y - GRIP_SPREAD, TOP + 0.30)
    plan.move(2.0, "over to the brick", right=(park_right, FINGER_OPEN, BRICK_GRIP))

    def both(duration: float, label: str, front: float, dy: float, z: float, finger: float, depth: float = PAD_DEPTH,
             loaded: bool = False, grip: float = BRICK_GRIP):
        """Both arms at the brick's front edge; ``loaded``: holding it, so z is raised by the sag at that reach."""
        targets = {}
        for arm in ys:
            tcp = tcp_at_front(front, ys[arm] + dy, z, depth)
            if loaded:
                tcp = (tcp[0], tcp[1], tcp[2] + sag(tcp[0]))
            targets[arm] = (tcp, finger, grip)
        plan.move(duration, label, **targets)

    tilted = SideGrip("z", CARRY_PITCH)
    both(2.5, "in front of the brick", front0 - 0.05, 0.0, zb, FINGER_OPEN)
    both(1.5, "reach the brick", front0, 0.0, zb, FINGER_OPEN)
    both(1.0, "grip", front0, 0.0, zb, FINGER_CLOSED)
    zc = BED + UNIT[2] / 2 + LIFT  # carried with its bottom 10 mm above the pocket's bed
    both(1.5, "lift", front0, 0.0, zc, FINGER_CLOSED, loaded=True, grip=tilted)
    both(1.0, "draw back", front0 - 0.03, 0.0, zc, FINGER_CLOSED, loaded=True, grip=tilted)
    both(2.5, "carry to the pocket", front0 - 0.03, -BRICK_Y, zc, FINGER_CLOSED, loaded=True, grip=tilted)
    both(1.0, "settle", front0 - 0.03, -BRICK_Y, zc, FINGER_CLOSED, loaded=True, grip=tilted)
    front_set = X_FACE + INSERT_DEPTH - UNIT[1]
    # Going in, the brick sinks 12-14 mm in the grip as the arms reach out (probes 45-47), so the hands rise
    # ENTRY_RISE over the move and it reaches the pocket at its carry height.
    both(3.5, "into the pocket", front_set, -BRICK_Y, zc + ENTRY_RISE, FINGER_CLOSED, loaded=True, grip=tilted)
    both(1.0, "level it", front_set, -BRICK_Y, zc + ENTRY_RISE, FINGER_CLOSED, loaded=True)
    both(1.0, "set it on the bed", front_set, -BRICK_Y, zc - LIFT - 0.004, FINGER_CLOSED, loaded=True)
    # Once the brick rests on the bed the arms no longer carry it, so the sag allowance would push the hands up and
    # jam the lower finger under the brick's front edge (probes 10-13: opening the fingers did not free it, and
    # backing off dragged it out). Take the allowance away while still holding, so the hands settle to their true
    # height with the bed carrying the brick, and only then let go.
    z_true = zc - LIFT  # the brick's centre, resting on the bed
    both(1.5, "take the weight off the hands", front_set, -BRICK_Y, z_true, FINGER_CLOSED)
    # Ease it in as a mason does: loosen the grip (not fully open), slide the hands back along the brick, grip
    # again and push it deeper, so the hold shrinks 35 -> 25 -> 15 -> 5 mm while the brick goes in 56 -> 86 mm.
    # The fingertips stay at the wall face, so the lower finger never meets the wall.
    front = front_set
    held = PAD_DEPTH
    for _ in range(EASE_STEPS):
        both(0.6, "loosen", front, -BRICK_Y, z_true, LOOSE, depth=held)
        held -= EASE_STEP
        both(0.8, "slide back along it", front, -BRICK_Y, z_true, LOOSE, depth=held)
        both(0.6, "grip again", front, -BRICK_Y, z_true, FINGER_CLOSED, depth=held)
        front += EASE_STEP
        both(1.2, "push it deeper", front + EASE_PUSH_PAST, -BRICK_Y, z_true, FINGER_CLOSED, depth=held)
    both(0.8, "let go", front, -BRICK_Y, z_true, FINGER_OPEN, depth=held)
    both(1.0, "wait for the fingers to open", front, -BRICK_Y, z_true, FINGER_OPEN, depth=held)
    both(1.5, "back off", front - BACK_OFF, -BRICK_Y, z_true, FINGER_OPEN, depth=held)
    both(0.8, "close the fingers", front - BACK_OFF, -BRICK_Y, z_true, FINGER_CLOSED, depth=held)
    front_set = front
    zp = BED + UNIT[2] / 2  # push at the middle of the brick's face
    both(1.0, "fingertips to the face", front_set - 0.005, -BRICK_Y, zp, FINGER_CLOSED, depth=0.0)
    both(3.0, "push it home", X_FACE + PUSH_PAST, -BRICK_Y, zp, FINGER_CLOSED, depth=0.0)
    both(1.5, "hold", X_FACE + PUSH_PAST, -BRICK_Y, zp, FINGER_CLOSED, depth=0.0)
    both(1.5, "step back", X_FACE - 0.06, -BRICK_Y, zp, FINGER_CLOSED, depth=0.0)
    return dict(plan.arms)


# --- the trowel's shank and blade: the task-file API has single boxes only -------------------------------------------
# yam_arena.scene.task_object(spec) turns each task object into an Arena object. Right after that module is imported,
# this wraps task_object, by object name: the trowel's spawner, after spawning the handle (the rigid body), adds the
# steel shank and blade as more colliders of the same body (as the accepted painting task does for its roller's wire
# frame). On a machine without the simulator nothing imports yam_arena.scene, and this does nothing.
_SCENE_MODULE = "yam_arena.scene"
PARTS_FOLDER = "trowel_parts"


def author_trowel(stage: Any, env_root: str) -> list[str]:
    """Add the shank and blade to the trowel's body under ``env_root``; the paths it made."""
    gf = importlib.import_module("pxr.Gf")
    sdf = importlib.import_module("pxr.Sdf")
    geom = importlib.import_module("pxr.UsdGeom")
    physics = importlib.import_module("pxr.UsdPhysics")
    shade = importlib.import_module("pxr.UsdShade")
    body = f"{env_root}/{TROWEL.name}"
    steel_path = f"{body}/{PARTS_FOLDER}/steel"
    material = shade.Material.Define(stage, steel_path)
    shader = shade.Shader.Define(stage, f"{steel_path}/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", sdf.ValueTypeNames.Color3f).Set(gf.Vec3f(*(_linear(c) for c in STEEL_RGB)))
    shader.CreateInput("metallic", sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("roughness", sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    made = [steel_path]
    for name, centre, size in TROWEL_PARTS:
        path = f"{body}/{PARTS_FOLDER}/{name}"
        cube = geom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xform = geom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(gf.Vec3d(*centre))
        xform.AddScaleOp().Set(gf.Vec3f(*size))
        physics.CollisionAPI.Apply(cube.GetPrim())
        shade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        made.append(path)
    return made


def _env_roots(prim: Any) -> list[str]:
    """Every environment folder holding the trowel, given the trowel prim just spawned (cloned copies included)."""
    stage = prim.GetStage()
    envs = prim.GetPath().GetParentPath().GetParentPath()
    return [str(env.GetPath()) for env in stage.GetPrimAtPath(envs).GetChildren() if env.GetChild(TROWEL.name).IsValid()]


def _with_parts(spawn: Callable[..., Any]) -> Callable[..., Any]:
    def spawn_and_build(prim_path: str, cfg: Any, *args: Any, **kwargs: Any) -> Any:
        prim = spawn(prim_path, cfg, *args, **kwargs)
        try:
            for root in _env_roots(prim):
                author_trowel(prim.GetStage(), root)
        except Exception as error:  # a trowel without its blade spreads nothing; scoring carries on
            _warn_once("trowel", f"could not build the trowel ({type(error).__name__}: {error})")
        return prim

    spawn_and_build.trowel_parts = True  # type: ignore[attr-defined]
    return spawn_and_build


def _spawner_cfgs(item: object) -> Iterator[Any]:
    """The spawner configs an Arena object holds (one level deep): objects with a callable ``func`` and a ``size``."""
    seen: set[int] = set()
    stack = [item, *vars(item).values()] if hasattr(item, "__dict__") else [item]
    for candidate in stack:
        for value in (candidate, getattr(candidate, "spawn", None), getattr(candidate, "spawner_cfg", None)):
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            if callable(getattr(value, "func", None)) and hasattr(value, "size"):
                yield value


def build_trowel(scene: Any) -> bool:
    """Wrap ``scene.task_object`` so the trowel comes out with its blade; True if it wrapped it now. It dispatches on
    the object's name, so a second load of this file is served by the first wrapper."""
    original = getattr(scene, "task_object", None)
    if original is None or getattr(original, "trowel_parts", False):
        return False

    def task_object(spec: Any) -> Any:
        made = original(spec)
        if getattr(spec, "name", None) == TROWEL.name:
            found = 0
            for cfg in _spawner_cfgs(made):
                found += 1
                if not getattr(cfg.func, "trowel_parts", False):
                    cfg.func = _with_parts(cfg.func)
            if not found:
                _warn_once("trowel", "found no spawner for the trowel; it will have no blade")
        return made

    task_object.trowel_parts = True  # type: ignore[attr-defined]
    scene.task_object = task_object
    return True


class _SceneWatcher(importlib.abc.MetaPathFinder):
    """Waits for yam_arena.scene to be imported, lets it load normally, then applies build_trowel. It answers for
    that one module name only, and removes itself once the module has loaded."""

    marker = "replace_brick.scene_watcher"

    def __init__(self) -> None:
        self._looking = False

    def find_spec(self, fullname: str, path: Sequence[str] | None,
                  target: types.ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname != _SCENE_MODULE or self._looking:
            return None
        self._looking = True  # find_spec below asks every finder again, this one included
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._looking = False
        if spec is not None and spec.loader is not None:
            spec.loader = _BuildAfterLoad(spec.loader, self)
        return spec

    def disarm(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)


class _BuildAfterLoad(importlib.abc.Loader):
    """The module's own loader, plus build_trowel once the module has run."""

    def __init__(self, inner: importlib.abc.Loader, watcher: _SceneWatcher) -> None:
        self._inner = inner
        self._watcher = watcher

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return self._inner.create_module(spec)

    def exec_module(self, module: types.ModuleType) -> None:
        self._inner.exec_module(module)
        build_trowel(module)
        self._watcher.disarm()

    def __getattr__(self, name: str) -> Any:  # get_source and friends, for tracebacks
        if name.startswith("__") or name in ("_inner", "_watcher"):  # a copy made without __init__
            raise AttributeError(name)
        return getattr(self._inner, name)


def install_trowel() -> None:
    """Wrap the scene builder now if it is already imported; otherwise wait for its import (once)."""
    scene = sys.modules.get(_SCENE_MODULE)
    if scene is not None:
        build_trowel(scene)
        return
    if not any(getattr(finder, "marker", None) == _SceneWatcher.marker for finder in sys.meta_path):
        sys.meta_path.insert(0, _SceneWatcher())


def _simulator_present() -> bool:
    try:
        return importlib.util.find_spec("isaaclab") is not None
    except (ImportError, ValueError):
        return False


if _simulator_present():
    try:
        install_trowel()
    except Exception as error:  # never break loading the task over the hook; the trowel then has no blade
        _warn_once("trowel", f"could not install the trowel hook ({type(error).__name__}: {error})")
