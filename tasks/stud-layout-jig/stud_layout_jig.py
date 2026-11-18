"""On the station table: a short wall section to frame on a framing table, three loose 2x4 studs and two cordless
impact drivers.

Carpenters frame walls lying flat on the deck or on a framing table: the plates are laid out with a pencil mark at
every stud, each stud is set on edge between the plates on its marks, and it is fastened through the plate into its
end. Here two 2x4 plates stand on edge on a 3/4-inch plywood board, a stud's length plus 6 mm apart: the bottom plate
on the arms' side, the top plate on the far side. Both carry layout marks for three studs, 16 inches on centre, and
the table has a pair of pine stops beside each mark at the top plate (3/4 x 1-1/2 x 6 inches, rounded over toward
the gap, 3 mm of clearance each side of a stud). At each mark a #10 x 3-inch screw with a #2 square-drive head sits
in a clearance hole through the bottom plate, its tip flush with the plate's inner face and its head standing
1-1/2 inches proud.

The three studs are surfaced 2x4s (1-1/2 x 3-1/2 inches actual) cut to 12 inches, about 0.47 kg. They start standing
on edge on the table beside the framing table, a little off square: one on the left, two on the right. A compact
12 V impact driver for each hand lies on its side on a rest in front of the table.

Goal: stand each stud on edge between the plates on its marks, then drive the screw at its mark through the bottom
plate into the stud's end until the head is flush.

Fidelity notes. The lumber, plywood, plates and stops are boxes at their real sizes; the screws and the drivers are
built from boxes and cylinders (yam_parts). Lumber density is taken as about 450 kg/m^3 (spruce-pine-fir). The
platform has no tilted or curved shapes, so each stop's rounded top is a row of overlapping spheres on a lower box;
the spheres also bulge about 19 mm past each stop's outer face, away from its gap. The screws are our model
(screw_runtime), not PhysX threads, but driving one takes the real work: the bit's tip touching the head and pressed
into it with at least 20 N, and a pad pressing the trigger with at least 20 N, both read from PhysX contacts. The
trigger's travel is not simulated. Left out: wood grain, the eased edges of real dimensional lumber, the second
screw a framer would drive at each stud end, and the top plate's screws (driven from the table's far side, out of
the arms' reach). Every free object has the platform's fixed friction (about 1.2), higher than wood on wood
(about 0.4-0.5).
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import screw_runtime as SR  # noqa: E402
import yam_parts  # noqa: E402
from yam_parts import Part, box, cyl  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, GRIPPER, PAD_TIP_PAST_TCP, SHOULDER, WRIST  # noqa: E402
import yam_arena.demo as motion  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.urdf import axis_angle_matrix  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Sphere  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INCH = 0.0254

# -- lumber (ALSC PS 20 dressed sizes) --------------------------------------------------
STUD_THICK = 1.5 * INCH  # 38.1 mm, across the slot (y when seated)
STUD_DEPTH = 3.5 * INCH  # 88.9 mm, vertical when on edge
STUD_LENGTH = 12.0 * INCH  # 304.8 mm cripple studs, along the slot (x when seated); short, so every grip on them
# is near enough the arms to hold them firmly
WOOD_DENSITY = 450.0  # kg/m^3, spruce-pine-fir near 12 % moisture
STUD_MASS = WOOD_DENSITY * STUD_THICK * STUD_DEPTH * STUD_LENGTH  # ~0.46 kg
STUD_COLOR = (0.87, 0.76, 0.56)

# -- jig and plates --------------------------------------------------------------------
PLY_THICK = 0.75 * INCH  # 19.05 mm
CLEAT_W = 0.75 * INCH  # 19.05 mm across y
CLEAT_H = 1.5 * INCH  # 38.1 mm tall, to the top of the rounded cap
CLEAT_L = 6.0 * INCH  # 152.4 mm along x
ROUND_R = CLEAT_W  # roundover radius: the cap curves from the cleat's outer edge down to its slot-side face
CLEAT_BOX_H = CLEAT_H - ROUND_R  # the square lower part of the cleat, 19.05 mm
ROUND_SPHERES = 16  # spheres per cleat; ~9 mm apart, so the ridges between them are ~0.5 mm
# 3 mm per side: at 1.5 mm a stud cocked more than ~1.1 degrees touched both cleats and wedged when pushed; at 3 mm
# that takes ~2.3 degrees.
SLOT_CLEARANCE = 0.003  # per side
SLOT_W = STUD_THICK + 2 * SLOT_CLEARANCE  # 44.1 mm
ON_CENTRE = 16.0 * INCH  # 406.4 mm
SLOT_Y = (-ON_CENTRE, 0.0, ON_CENTRE)  # stud centre lines, right to left

PLATE_T = STUD_THICK  # a 2x4 plate on edge: 38.1 mm along x
PLATE_H = STUD_DEPTH  # 88.9 mm tall
PLATE_LEN = 2 * ON_CENTRE + STUD_THICK  # 33-1/2 in: flush with the outer faces of the end studs
END_CLEARANCE = 0.006  # the plates stand this much more than a stud length apart (3 mm at each end when centred),
# so a stud lowered between them clears both; the screw then draws it tight to the bottom plate
# The bottom plate is as close to the arms as the screwdriver allows: driving a screw whose head stands 1-1/2 in
# proud puts the hand on the impact driver's handle about 0.11 m nearer the arms than the head (checked offline by
# IK), and the driver's motor housing about 0.17 m nearer; 30 mm farther out keeps the housing clear of the arms' bases
# at the outer screws.
BOTTOM_PLATE_INNER_X = 0.256  # a stud tight to the bottom plate has its near end here
BOTTOM_PLATE_OUTER_X = BOTTOM_PLATE_INNER_X - PLATE_T
TOP_PLATE_FACE_X = BOTTOM_PLATE_INNER_X + STUD_LENGTH + END_CLEARANCE  # the top plate's inner face
PLATE_COLOR = (0.70, 0.56, 0.36)
MARK_W, MARK_H = 0.0015, 0.0005  # pencil lines on the plates' top edges, at each stud's two faces
MARK_COLOR = (0.15, 0.15, 0.15)

BASE_TOP = TOP + PLY_THICK
CLEAT_NEAR_X = TOP_PLATE_FACE_X - CLEAT_L
JIG_NEAR_X = BOTTOM_PLATE_OUTER_X - 0.020
JIG_FAR_X = TOP_PLATE_FACE_X + PLATE_T
JIG_HALF_Y = max(abs(y) for y in SLOT_Y) + SLOT_W / 2 + 2 * CLEAT_W + 0.015
PLY_COLOR = (0.80, 0.66, 0.44)
CLEAT_COLOR = (0.62, 0.47, 0.28)

# -- screws: #10 x 3 in construction screws, started through the bottom plate ----------
SCREW_LEN = 3.0 * INCH  # 76.2 mm, head underside to tip
SCREW_Z = BASE_TOP + STUD_DEPTH / 2  # on the stud's end, at mid-depth
SHANK_R = 0.0024  # #10: 4.8 mm
HEAD_R, HEAD_T = 0.0049, 0.003  # 9.8 mm head (estimate), 3 mm thick
SCREW_PROUD = SCREW_LEN - PLATE_T  # the started screw's tip is flush with the plate's inner face: 38.1 mm proud
SCREW_HEAD_X = BOTTOM_PLATE_INNER_X - SCREW_LEN  # the head's underside when started
SCREW_SEATED_X = BOTTOM_PLATE_OUTER_X  # the head's underside when driven home
HOLE_HALF = 0.003  # the plate's pilot hole, 6 mm square in the simulation
STEEL = (0.65, 0.65, 0.68)
SCREW_COLOR = (0.10, 0.10, 0.11)  # black-coated structural framing screws (they show against the pine plates)

# -- impact driver: a compact 12 V cordless impact driver with a 2 Ah battery (estimates from typical models) ------
# Frame: origin at the bit's tip, +x the bit's direction, the pistol grip toward +y (the driver is used, and lies,
# on its side, so the grip and battery stay clear of the plywood with the bit at a stud's mid-depth).
BIT_LEN, BIT_R = 0.025, 0.0032  # 1/4 in hex insert bit, #2 square tip, 25 mm out of the chuck
CHUCK = (0.020, 0.0105)  # length, radius
HOUSING = (0.120, 0.056, 0.056)  # motor and hammer housing behind the chuck
HANDLE_DEPTH, HANDLE_LEN, HANDLE_T = 0.045, 0.100, 0.034  # front to back, along the grip, across it
BATTERY = (0.105, 0.044, 0.072)
HOUSING_X = -(BIT_LEN + CHUCK[0] + HOUSING[0] / 2)
HANDLE_X = -(BIT_LEN + CHUCK[0] + 0.065)  # the grip hangs from the rear of the housing
HANDLE_Y = HOUSING[1] / 2 + HANDLE_LEN / 2  # the grip's middle
# The trigger: a block standing proud of the grip's front face, where the hand closes on the grip (at its balance
# point). The robot's pads close on its face and on the grip's back.
# The trigger: out of the grip, along it, across it. Broad, so the pad on it seats as firmly as on the grip's back
# (on a 22 x 12 mm trigger the driver twisted in the hand when knocked, and fell).
TRIGGER_PROUD, TRIGGER_LEN, TRIGGER_W = 0.006, 0.040, 0.028
TRIGGER_Y = 0.071  # along the grip: the driver's balance point
TRIGGER_FACE_X = HANDLE_X + HANDLE_DEPTH / 2 + TRIGGER_PROUD
GRIP_SPAN = HANDLE_DEPTH + TRIGGER_PROUD  # the trigger's face to the grip's back: what the pads close on
DRIVER_MASS = 0.95  # kg, with its battery
REST_H = BATTERY[2] / 2 - HOUSING[2] / 2  # the rest under the housing, so it lies level on housing and battery
BLACK, ORANGE, BLUE, RED = (0.08, 0.08, 0.08), (0.90, 0.40, 0.10), (0.20, 0.35, 0.75), (0.75, 0.08, 0.08)

SEATED_X = BOTTOM_PLATE_INNER_X + STUD_LENGTH / 2  # a stud's centre, tight to the bottom plate
SEATED_Z = BASE_TOP + STUD_DEPTH / 2


def _bottom_plate():
    """The bottom plate, with a pilot hole at each mark for its screw (the screws' shanks pass through it)."""
    cx = BOTTOM_PLATE_OUTER_X + PLATE_T / 2
    half = PLATE_LEN / 2
    shapes = []
    edges = [-half]
    for y in SLOT_Y:
        edges += [y - HOLE_HALF, y + HOLE_HALF]
    edges.append(half)
    for y0, y1 in zip(edges[0::2], edges[1::2]):  # solid stretches between the holes
        shapes.append(box((0.0, (y0 + y1) / 2, 0.0), (PLATE_T, y1 - y0, PLATE_H)))
    for y in SLOT_Y:  # above and below each hole
        h = PLATE_H / 2 - HOLE_HALF
        for sz in (-1, 1):
            shapes.append(box((0.0, y, sz * (HOLE_HALF + h / 2)), (PLATE_T, 2 * HOLE_HALF, h)))
    return Part("bottom_plate", (cx, 0.0, BASE_TOP + PLATE_H / 2), shapes, color=PLATE_COLOR)


def _marks():
    parts = []
    for plate, cx in (("bottom", BOTTOM_PLATE_OUTER_X + PLATE_T / 2), ("top", TOP_PLATE_FACE_X + PLATE_T / 2)):
        for i, y in enumerate(SLOT_Y, start=1):
            for side, sign in (("a", -1.0), ("b", 1.0)):
                parts.append(
                    Box(
                        f"mark_{plate}_{i}{side}",
                        pos=(cx, y + sign * STUD_THICK / 2, BASE_TOP + PLATE_H + MARK_H / 2),
                        size=(PLATE_T, MARK_W, MARK_H),
                        color=MARK_COLOR,
                    )
                )
    return parts


def _jig():
    parts = [
        Box(
            "jig_base",
            pos=((JIG_NEAR_X + JIG_FAR_X) / 2, 0.0, TOP + PLY_THICK / 2),
            size=(JIG_FAR_X - JIG_NEAR_X, 2 * JIG_HALF_Y, PLY_THICK),
            color=PLY_COLOR,
        ),
        _bottom_plate(),
        Box(
            "top_plate",
            pos=(TOP_PLATE_FACE_X + PLATE_T / 2, 0.0, BASE_TOP + PLATE_H / 2),
            size=(PLATE_T, PLATE_LEN, PLATE_H),
            color=PLATE_COLOR,
        ),
        *_marks(),
    ]
    cleat_x = TOP_PLATE_FACE_X - CLEAT_L / 2
    box_offset = SLOT_W / 2 + CLEAT_W / 2
    # Sphere centres run along the cleat's outer top edge at the top of the square part, so each sphere meets the
    # slot-side face tangentially at that height and the slot keeps its full width all the way down.
    round_offset = SLOT_W / 2 + CLEAT_W
    round_z = BASE_TOP + CLEAT_BOX_H
    round_xs = np.linspace(CLEAT_NEAR_X, TOP_PLATE_FACE_X - ROUND_R, ROUND_SPHERES)
    for i, y in enumerate(SLOT_Y, start=1):
        for side, sign in (("a", -1.0), ("b", 1.0)):
            parts.append(
                Box(
                    f"cleat_{i}{side}",
                    pos=(cleat_x, y + sign * box_offset, BASE_TOP + CLEAT_BOX_H / 2),
                    size=(CLEAT_L, CLEAT_W, CLEAT_BOX_H),
                    color=CLEAT_COLOR,
                )
            )
            for k, rx in enumerate(round_xs):
                parts.append(
                    Sphere(
                        f"cleat_{i}{side}_round_{k:02d}",
                        pos=(float(rx), y + sign * round_offset, round_z),
                        radius=ROUND_R,
                        color=CLEAT_COLOR,
                    )
                )
    return parts


def make_screw(i, y):
    """A started screw, its frame at the head's underside on the screw's axis, +x its direction of travel. Only
    the head and the shank between the head and the plate are colliders; the rest of the shank (in the plate and,
    once driven, in the stud) is not simulated. screw_runtime holds it on its axis and moves it as it is driven."""
    shapes = [
        cyl((-HEAD_T / 2, 0.0, 0.0), HEAD_R, HEAD_T, "x", STEEL),
        cyl((SCREW_PROUD / 2, 0.0, 0.0), SHANK_R, SCREW_PROUD, "x", STEEL),
    ]
    return Part(f"screw_{i}", (SCREW_HEAD_X, y, SCREW_Z), shapes, mass=0.02, color=SCREW_COLOR)


def make_driver(pos, name="driver_right", side=1.0):
    """An impact driver, lying on its side: bit, chuck, housing, pistol grip with its trigger, and battery. ``side``
    +1 lays it with its grip toward +y (the robot's left), -1 toward -y."""
    s = [cyl((-BIT_LEN / 2, 0.0, 0.0), BIT_R, BIT_LEN, "x", STEEL),
         cyl((-(BIT_LEN + CHUCK[0] / 2), 0.0, 0.0), CHUCK[1], CHUCK[0], "x", BLACK),
         box((HOUSING_X, 0.0, 0.0), HOUSING, ORANGE),
         box((HANDLE_X, side * HANDLE_Y, 0.0), (HANDLE_DEPTH, HANDLE_LEN, HANDLE_T), BLACK),
         box((HANDLE_X + 0.005, side * (HOUSING[1] / 2 + HANDLE_LEN + BATTERY[1] / 2), 0.0), BATTERY, BLACK),
         box((TRIGGER_FACE_X - TRIGGER_PROUD / 2, side * TRIGGER_Y, 0.0), (TRIGGER_PROUD, TRIGGER_LEN, TRIGGER_W),
             RED)]
    return Part(name, pos, s, mass=DRIVER_MASS, color=ORANGE, friction=(0.60, 0.50), contact_offset=0.001,
                grip=3)


# The hand holds the grip level with the driver's centre of mass (uniform density, as PhysX takes it), so the
# driver hangs balanced in a sideways grip instead of rolling in the pads; and midway between the trigger's face and
# the grip's back, where the pads close.
DRIVER_GRIP = (TRIGGER_FACE_X - GRIP_SPAN / 2, float(make_driver((0.0, 0.0, 0.0)).volume_centroid()[1]), 0.0)
# Both drivers lie, and are held, with their grips toward +y. (Mirrored, toward -y, the right driver would keep its
# battery clear of the right arm at the right screw, but the right hand could then reach that screw only turned
# round, and its rest and the middle screw only the other way: it cannot turn round holding the driver.)
DRIVER_SIDE = {"left": 1.0, "right": 1.0}


def driver_grip(arm):
    """Where the hand holds that arm's driver, in the driver's frame."""
    return (DRIVER_GRIP[0], DRIVER_SIDE[arm] * DRIVER_GRIP[1], DRIVER_GRIP[2])


# Each hand has its own driver, on its own rest (bit tips given): neither arm can reach the other's driving poses
# with the same hold on the grip (checked offline by IK), so they cannot share one.
DRIVER_STARTS = {
    "left": (0.10 - HANDLE_X, 0.44 - DRIVER_GRIP[1], TOP + BATTERY[2] / 2),  # the grip at (0.10, 0.44), clear
    # of the left arm's base (at 0.06, 0.36 the driver lay against it and the hand could not reach round the grip)
    "right": (0.05 - HANDLE_X, 0.0 + DRIVER_GRIP[1], TOP + BATTERY[2] / 2),  # the grip at (0.05, 0.00)
}
DRIVER_NAMES = {"left": "driver_left", "right": "driver_right"}


def make_rest(arm):
    """A low block under a driver's housing, so it lies level on its side on the housing and the battery."""
    x, y, _ = DRIVER_STARTS[arm]
    return Part(f"rest_{arm}", (x + HOUSING_X, y, TOP), [box((0.0, 0.0, REST_H / 2), (HOUSING[0] - 0.02, 0.040, REST_H))],
                color=BLUE)


# Starting layout, all on edge on the table beside the jig and turned a little off square (x, y, yaw, bottom
# height). The table ends at y = +-0.65 m; the two on the right stand ~31 mm apart, room for a finger between them.
START = {
    "stud_1": (0.30, 0.585, -0.06, TOP),
    "stud_2": (0.30, -0.53, 0.03, TOP),
    "stud_3": (0.30, -0.61, -0.03, TOP),
}
STUDS = tuple(
    Box(
        name,
        pos=(x, y, bottom + STUD_DEPTH / 2),
        size=(STUD_LENGTH, STUD_THICK, STUD_DEPTH),
        mass=STUD_MASS,
        color=STUD_COLOR,
        yaw=yaw,
    )
    for name, (x, y, yaw, bottom) in START.items()
)
SCREWS = tuple(make_screw(i, y) for i, y in enumerate(SLOT_Y, start=1))
DRIVERS = tuple(make_driver(DRIVER_STARTS[arm], DRIVER_NAMES[arm], DRIVER_SIDE[arm]) for arm in ("left", "right"))
OBJECTS = [*_jig(), *SCREWS, make_rest("left"), make_rest("right"), *STUDS, *DRIVERS]
PRESSED_PARTS = tuple(DRIVER_NAMES.values())  # the pads squeeze its grip; advisory offline, as in the lever-set task

# The screw model's geometry (screw_runtime).
SR.SCREW_STARTS = tuple((SCREW_HEAD_X, y, SCREW_Z) for y in SLOT_Y)
SR.SCREW_PROUD = SCREW_PROUD
SR.PLATE_INNER_X = BOTTOM_PLATE_INNER_X
SR.STUDS = tuple(s.name for s in STUDS)
SR.SCREWS = tuple(s.name for s in SCREWS)
SR.STUD_HALF = (STUD_LENGTH / 2, STUD_THICK / 2, STUD_DEPTH / 2)
SR.HEAD_T = HEAD_T
SR.BIT_R = BIT_R
SR.TRIGGER_FACE = {  # each driver's trigger face in its own frame
    DRIVER_NAMES[arm]: (TRIGGER_FACE_X, tuple(sorted(DRIVER_SIDE[arm] * (TRIGGER_Y + d * TRIGGER_LEN / 2) for d in (-1, 1))),
                        (-TRIGGER_W / 2, TRIGGER_W / 2))
    for arm in ("left", "right")
}
SR.DRIVERS = tuple(DRIVER_NAMES.values())
SR.NAMES = (*SR.DRIVERS, *SR.STUDS, *SR.SCREWS)
SR.MODEL.clear()
SR.prepare()  # PhysX contact processing on, for the model's contact reads


# At spawn (GPU host only): the drivers report their contacts (the bit on a screw's head, a pad on the trigger), and
# the screws are kinematic, so a head stops the bit as a screw held by its thread does, and the model moves them.
_original_spawn_part = getattr(yam_parts._spawn_part, "_studs_original", yam_parts._spawn_part)


def _spawn_part(prim_path, cfg, *args, **kwargs):
    prim = _original_spawn_part(prim_path, cfg, *args, **kwargs)
    name = getattr(cfg, "part_name", "")
    if name in SR.DRIVERS or name in SR.SCREWS:
        from pxr import PhysxSchema, UsdPhysics

        if name in SR.DRIVERS:
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        else:
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
    return prim


_spawn_part._studs_original = _original_spawn_part
yam_parts._spawn_part = _spawn_part
yam_parts._CLONED_SPAWN_PART = None  # cloned from _spawn_part when the scene is built

# -- scoring -------------------------------------------------------------------------------
# Each stud scores up to 1/3, on its own; each mark (slot) credits at most one stud.
PLACED_CREDIT = 0.10  # on edge between the plates, on its marks, square
CAUGHT_CREDIT = 0.10  # its screw has gone into the stud's end
FASTENED_CREDIT = 1.0 / 3.0 - PLACED_CREDIT - CAUGHT_CREDIT  # screw driven home, stud tight to the bottom plate
FLUSH_TOLERANCE = 0.001  # a screw is driven home when its head is within 1 mm of flush with the plate (a straightedge
# across the plate's face, or a 1 mm feeler gauge that will not go under the head's edge)
HEIGHT_TOLERANCE = 0.003  # bottom on the plywood
ANGLE_TOLERANCE = math.radians(3)  # upright on edge, and square to the plates
MARK_TOLERANCE = 0.004  # the stud's centre line from the centre of its marks (a tape measure)
TIGHT_TOLERANCE = 0.001  # the stud's end against the bottom plate (a 1 mm feeler gauge does not go in)
RELEASE_CLEAR = 0.030  # every hand at least this far from every stud for full credit
QUAT_NORM_TOLERANCE = 1e-3
OUTLINE_TOLERANCE = 0.002  # built box vs the spec, per axis
STUD_HALF = np.array([STUD_LENGTH, STUD_THICK, STUD_DEPTH]) / 2
TEST_SCREWS = None  # offline fixtures set the screws' state: [(caught stud or None, depth driven in m), ...]


def _finite(state) -> bool:
    fields = (state.pos, state.quat_xyzw, state.lin_vel, state.ang_vel, state.low, state.high)
    if not all(np.all(np.isfinite(f)) for f in fields):
        return False
    return abs(float(np.linalg.norm(state.quat_xyzw)) - 1.0) <= QUAT_NORM_TOLERANCE


def _rotation(state) -> np.ndarray:
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _valid(objects) -> bool:
    """Fail-safe: every stud is present with finite values and a unit quaternion, and its built outline (the
    world-aligned box the simulator reports) matches a 1-1/2 x 3-1/2 x 12-inch box at its pose."""
    for stud in STUDS:
        state = objects.get(stud.name)
        if state is None or not _finite(state):
            return False
        expected = 2 * np.abs(_rotation(state)) @ STUD_HALF
        built = np.asarray(state.high, dtype=float) - np.asarray(state.low, dtype=float)
        if np.any(np.abs(built - expected) > OUTLINE_TOLERANCE):
            return False
    return True


def _axes(state):
    """The stud's long (local x) and depth (local z) axes in the world frame."""
    rot = _rotation(state)
    return rot[:, 0], rot[:, 2]


def _slot_of(state):
    """The index of the mark this stud stands on, on edge between the plates and square to them, or None."""
    long_axis, depth_axis = _axes(state)
    upright = abs(depth_axis[2]) >= math.cos(ANGLE_TOLERANCE)
    square = abs(long_axis[0]) >= math.cos(ANGLE_TOLERANCE)
    on_base = abs(float(state.low[2]) - BASE_TOP) <= HEIGHT_TOLERANCE
    between = float(state.low[0]) >= BOTTOM_PLATE_INNER_X - 0.001 and float(state.high[0]) <= TOP_PLATE_FACE_X + 0.001
    if not (upright and square and on_base and between):
        return None
    for index, slot_y in enumerate(SLOT_Y):
        if abs(float(state.pos[1]) - slot_y) <= MARK_TOLERANCE:
            return index
    return None


def _tight(state) -> bool:
    return float(state.low[0]) - BOTTOM_PLATE_INNER_X <= TIGHT_TOLERANCE


def _screws():
    """Per screw (by mark): the stud it caught (or None) and how far it has been driven in (m)."""
    if TEST_SCREWS is not None:
        return TEST_SCREWS
    m = SR.MODEL
    return [(m.caught[i], m.advance[i]) for i in range(len(SLOT_Y))]


def _home(depth: float) -> bool:
    """The screw's head within FLUSH_TOLERANCE of flush with the plate."""
    return depth >= SCREW_PROUD - FLUSH_TOLERANCE


def _results(objects):
    """(placed, caught, fastened) per stud, each mark credited to at most one stud."""
    screws = _screws()
    used = set()
    out = []
    for stud in STUDS:
        state = objects[stud.name]
        slot = _slot_of(state)
        if slot is None or slot in used:
            out.append((False, False, False))
            continue
        used.add(slot)
        caught_by, depth = screws[slot]
        caught = caught_by == stud.name
        out.append((True, caught, caught and _home(depth) and _tight(state)))
    return out


def _hands():
    """Each arm's TCP on the simulator, or None offline."""
    env = _RUN["env"] if "_RUN" in globals() else None
    if env is None:
        return None
    try:
        base = env.unwrapped
        origin = SR._np(base.scene.env_origins)[0][:3]
        return [tcp for tcp, _, _ in SR._grips(base, origin).values()]
    except Exception as e:  # noqa: BLE001
        print(f"HANDS_READ failed: {e!r}", flush=True)
        return None


def _released(objects) -> bool:
    hands = _hands()
    if hands is None:
        return True  # offline the arms are not known: assumed clear
    for stud in STUDS:
        st = objects[stud.name]
        low, high = np.asarray(st.low), np.asarray(st.high)
        for tcp in hands:
            if np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) < RELEASE_CLEAR:
                return False
    return True


def score(objects):
    """Per stud: 0.10 placed on its marks, 0.10 more once its screw has caught it, 0.133 more once the screw is
    seated with the stud tight to the bottom plate (1/3 per stud)."""
    if not _valid(objects):
        return 0.0
    total = sum(PLACED_CREDIT * p + CAUGHT_CREDIT * c + FASTENED_CREDIT * f for p, c, f in _results(objects))
    return float(min(1.0, total))


def success(objects):
    """All three studs on their marks and fastened, and the hands clear of them."""
    return _valid(objects) and all(f for _, _, f in _results(objects)) and _released(objects)


def distance(objects):
    """Mean distance from each stud's centre to the nearest fastened position, in metres."""
    if not _valid(objects):
        return float("inf")
    targets = [np.array([SEATED_X, y, SEATED_Z]) for y in SLOT_Y]
    return float(
        np.mean([min(np.linalg.norm(objects[s.name].pos - t) for t in targets) for s in STUDS])
    )


# -- scripted demo ------------------------------------------------------------------------
# With the hand straight down, the arms cannot lift a stud clear of the plates (checked offline by IK: the hand
# reaches at most 0.11-0.17 m above the table over the slots, and needs ~0.19 m). Tipping the fingertips 30 degrees
# toward the top plate, about the fingers' closing axis, lets it reach 0.23 m or more everywhere the demo goes, the
# centre slot's stud held at its middle included (15 degrees reached it only 113 mm off its middle, where the stud
# swung down in the pads on the simulator). The
# standard waypoint only turns a straight-down hand about the vertical, so, as the lever-set task does, the demo's grip_rotation
# is replaced by one that also takes a full rotation.
TILT = math.radians(30.0)
GRIP_DEPTH = 0.050  # pad tips this far below the stud's top edge (deep, for a firm hold on its faces)
PLATE_CLEAR = 0.020  # a carried stud's bottom clears the plates' top edges by this much
DROP_GAP = 0.005  # the stud is let go with its bottom this far above the cleat tops
PLACE_X = BOTTOM_PLATE_INNER_X + END_CLEARANCE / 2 + STUD_LENGTH / 2  # centred between the plates
ASSIGNMENT = {  # stud -> (arm, slot index), in order per arm
    "stud_1": ("left", 2),
    "stud_2": ("right", 1),  # the right arm seats the centre stud first, so it never carries a stud over a seated one
    "stud_3": ("right", 0),
}


class GripOrientation(float):
    """A grip's closing azimuth (what a plain waypoint's yaw holds) carrying the full rotation for IK, as in the lever-set task."""

    def __new__(cls, matrix):
        closure = np.asarray(matrix, float)[:, 0]
        obj = super().__new__(cls, math.atan2(-closure[0], closure[1]))
        obj.matrix = np.asarray(matrix, float).copy()
        return obj


_original_grip_rotation = getattr(motion.grip_rotation, "_original", motion.grip_rotation)


def grip_rotation(yaw=0.0):
    matrix = getattr(yaw, "matrix", None)
    if isinstance(matrix, np.ndarray) and matrix.shape == (3, 3):
        return matrix
    return _original_grip_rotation(yaw)


grip_rotation._original = _original_grip_rotation
motion.grip_rotation = grip_rotation


def tilted(yaw: float) -> GripOrientation:
    """A downward grip turned ``yaw`` about the vertical, its fingertips tipped TILT along the stud (toward +x when
    yaw is 0); the fingers still close across the stud's thickness."""
    z_axis, y_axis = np.array((0.0, 0.0, 1.0)), np.array((0.0, 1.0, 0.0))
    R = axis_angle_matrix(z_axis, yaw) @ axis_angle_matrix(y_axis, -TILT) @ _original_grip_rotation(0.0)
    return GripOrientation(R)


def _tcp(grip_xy, yaw: float, bottom_z: float):
    """The TCP that puts the pad tips at ``grip_xy`` (on the stud's centre line), GRIP_DEPTH below the top edge of a
    stud whose bottom is at ``bottom_z``. The tilt moves the tips TILT-wise ahead of the TCP along the stud."""
    ahead = PAD_TIP_PAST_TCP * math.sin(TILT)
    x = grip_xy[0] - ahead * math.cos(yaw)
    y = grip_xy[1] - ahead * math.sin(yaw)
    return (x, y, bottom_z + STUD_DEPTH - GRIP_DEPTH + PAD_TIP_PAST_TCP * math.cos(TILT))


PLATE_TOP = BASE_TOP + PLATE_H
TRAVEL_BOTTOM = PLATE_TOP + PLATE_CLEAR
ENTRY_BOTTOM = PLATE_TOP + 0.005  # just above the plates' top edges: the last read before going down between them
DROP_BOTTOM = BASE_TOP + CLEAT_H + DROP_GAP  # just above the cleats: the last read before going down into the slot
# Let go of in mid-air, a stud was dragged ~15 mm sideways by the opening pads (one lets go before the other) and
# landed on a cleat. So the stud is lowered all the way into its slot, the cleats either side of it, before the
# fingers open.
RELEASE_BOTTOM = BASE_TOP + 0.002
# The hand holds each stud this far from its middle toward the bottom plate, so that with the stud in its slot the
# tilted fingertips stay short of the cleats (which start 3 mm from the stud's middle) as they open.
GRIP_BACK = 0.025
# Carried in one hand, a stud sags and tips in the grip (on the simulator: ~30 mm low, 12-18 mm short of the top
# plate, 12-17 mm to the right and tipped ~6 degrees), more than the 3 mm the plates and cleats leave. So, as the lever-set task's
# demo does, the hand reads the stud's pose over its slot and moves so the stud is square and where it should be:
# twice above the plates, once more just above them, and once between them before letting go. A camera sees the same.
MAX_SHIFT = 0.040  # m, the most the reads may move the hand from the plan
MAX_TURN = math.radians(15.0)


def _run_context():
    """Keeps the live environment where the demo can read the studs' poses (as the lever-set task's _run_context does)."""
    context = {"env": None}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run_studs"):
        return cls._rc_run_studs
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"] = self._env
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        SR.MODEL.clear()
        result = original_reset(self, *args, **kwargs)
        SR.MODEL.clear()  # whatever the settle steps did, every trial starts with the screws as started
        context["env"] = self._env
        try:
            SR.install(self._env)
        except Exception as e:  # noqa: BLE001
            print(f"SCREW_HOOK failed: {e!r}", flush=True)
        return result

    cls._rc_run_studs = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()


def _stud_now(name):
    """The stud's state on the simulator now, or None offline."""
    if _RUN["env"] is None:
        return None
    try:
        from yam_arena import scene

        return scene.object_states(_RUN["env"], [name])[name]
    except Exception as e:  # noqa: BLE001
        print(f"STUD_READ failed: {e!r}", flush=True)
        return None


def _turn_angle(R) -> float:
    return math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))


def _squaring(R_stud):
    """The rotation that turns the stud square to the plates and upright (either end may face the top plate)."""
    turned = axis_angle_matrix(np.array((0.0, 0.0, 1.0)), math.pi)
    fix = min((np.eye(3) @ R_stud.T, turned @ R_stud.T), key=_turn_angle)
    angle = _turn_angle(fix)
    return fix if angle <= MAX_TURN else ik.slerp_matrix(np.eye(3), fix, MAX_TURN / angle)


class Lazy(motion.Waypoint):
    """A waypoint whose target is worked out when the arm first reaches it, so it can use a read (as in the lever-set task)."""

    def __init__(self, duration, where, finger, label):
        super().__init__(duration, (0.0, 0.0, 0.0), finger, label, 0.0)
        self._where, self._resolved = where, None

    def _resolve(self):
        if self._resolved is None:
            p, R = self._where()
            self._resolved = (tuple(float(v) for v in p), GripOrientation(R))
        return self._resolved

    @property
    def pos(self):
        return self._resolve()[0]

    @pos.setter
    def pos(self, value):
        pass

    @property
    def yaw(self):
        return self._resolve()[1]

    @yaw.setter
    def yaw(self, value):
        pass


def _seat_waypoints(stud: Box, slot: int) -> list[Waypoint]:
    sx, sy, sz = stud.pos
    yaw = stud.yaw
    slot_y = SLOT_Y[slot]
    pick_xy = (sx - GRIP_BACK * math.cos(yaw), sy - GRIP_BACK * math.sin(yaw))  # near the middle, its balance point
    place_xy = (PLACE_X - GRIP_BACK, slot_y)
    bottom = sz - STUD_DEPTH / 2
    pick_up, pick = _tcp(pick_xy, yaw, TRAVEL_BOTTOM), _tcp(pick_xy, yaw, bottom)
    at_pick, at_place = tilted(yaw), tilted(0.0)
    fix = {"dp": np.zeros(3), "R": np.eye(3)}  # the reads' correction to the planned hand pose, for this stud

    def at(stud_bottom):
        """The hand pose that holds the stud over its slot with its bottom at ``stud_bottom``, read corrections in."""
        return lambda: (np.asarray(_tcp(place_xy, 0.0, stud_bottom)) + fix["dp"], fix["R"] @ at_place.matrix)

    def read(stud_bottom, label):
        def where():
            plan = np.asarray(_tcp(place_xy, 0.0, stud_bottom))
            hand = plan + fix["dp"]
            st = _stud_now(stud.name)
            if st is not None:
                centre = np.asarray(st.pos, float)
                turn = _squaring(_rotation(st))
                goal = np.array((PLACE_X, slot_y, stud_bottom + STUD_DEPTH / 2))
                # turn the hand (and the stud in it) about the stud's centre, then move that centre onto the goal
                moved = goal + turn @ (hand - centre)
                fix["dp"] = np.clip(moved - plan, -MAX_SHIFT, MAX_SHIFT)
                fix["R"] = turn @ fix["R"]
                err = (centre - goal) * 1000
                print(f"STUD_READ {stud.name} {label}: off by x {err[0]:+.1f} y {err[1]:+.1f} z {err[2]:+.1f} mm, "
                      f"turned {math.degrees(_turn_angle(turn)):.1f} deg", flush=True)
            return at(stud_bottom)()

        return where

    return [
        Waypoint(1.5, pick_up, FINGER_OPEN, "over the stud", at_pick),
        Waypoint(1.5, pick, FINGER_OPEN, "descend", at_pick),
        Waypoint(0.8, pick, FINGER_CLOSED, "close", at_pick),
        Waypoint(1.5, _tcp(pick_xy, yaw, bottom + 0.030), FINGER_CLOSED, "lift off the table", at_pick),
        Waypoint(1.5, pick_up, FINGER_CLOSED, "lift above the plates' height", at_pick),
        Waypoint(2.5, _tcp(place_xy, 0.0, TRAVEL_BOTTOM), FINGER_CLOSED, "carry over the plates and square up",
                 at_place),
        Waypoint(0.8, _tcp(place_xy, 0.0, TRAVEL_BOTTOM), FINGER_CLOSED, "settle", at_place),
        Lazy(1.0, read(TRAVEL_BOTTOM, "over the slot"), FINGER_CLOSED, "line the stud up over its slot"),
        Lazy(0.8, at(TRAVEL_BOTTOM), FINGER_CLOSED, "settle"),
        Lazy(0.8, read(TRAVEL_BOTTOM, "over the slot, again"), FINGER_CLOSED, "line it up again"),
        Lazy(1.5, at(ENTRY_BOTTOM), FINGER_CLOSED, "down to the plates' top edges"),
        Lazy(0.6, at(ENTRY_BOTTOM), FINGER_CLOSED, "settle"),
        Lazy(0.8, read(ENTRY_BOTTOM, "at the plates' edges"), FINGER_CLOSED, "line it up with the gap"),
        Lazy(2.0, at(DROP_BOTTOM), FINGER_CLOSED, "lower between the plates"),
        Lazy(0.6, at(DROP_BOTTOM), FINGER_CLOSED, "steady"),
        Lazy(0.6, read(DROP_BOTTOM, "between the plates"), FINGER_CLOSED, "line it up over the cleats"),
        Lazy(1.5, at(RELEASE_BOTTOM), FINGER_CLOSED, "lower it into the slot"),
        Lazy(0.5, at(RELEASE_BOTTOM), FINGER_CLOSED, "steady"),
        Lazy(0.8, at(RELEASE_BOTTOM), FINGER_OPEN, "let go"),
        Lazy(1.5, at(TRAVEL_BOTTOM), FINGER_OPEN, "up"),
    ]


# -- driving the screws ---------------------------------------------------------------------
# The hand holds the impact driver by its grip from above, one pad on the trigger's face and the other on the grip's
# back. Each finger presses with about 5 N per mm it is driven past the surface it touches (the gripper's stiffness,
# 5000 N/m, up to its 50 N limit). A light hold (3 mm, ~15 N) carries the driver without pulling the trigger
# (20 N); a squeeze (7 mm, ~35 N) pulls it. Pushing the bit into a head with F takes F/2 off the trigger pad and puts
# it on the back pad, so with the squeeze the trigger stays pulled, and the back pad within its limit, for pushes up
# to 30 N; the demo pushes 25 N (the screw needs 20 N). As with the studs, the hand reads where the bit is before
# going in, and lines it up with the screw.
FINGER_K = GRIPPER["stiffness"]
# each finger's position: the pads 2.8 mm past the trigger and the grip's back, ~14 N, under the trigger's 17 N and
# enough to carry the driver (the drops seen at 14 and 18 N were knocks against the arm's base, now avoided)
FINGER_HOLD = -(GRIP_SPAN / 2 - 0.0028)
# The squeeze: the pads 8 mm in, ~40 N. Pushing the bit loads the back pad and unloads the trigger by about half the
# push, and the finger's 50 N limit caps the back pad, so the trigger stays at 17 N or more for pushes up to 33 N
# (at 7 mm, ~35 N, a 26 N push left the trigger at 19.8 N, under the 20 N it then took, and the screw
# never turned).
FINGER_SQUEEZE = -(GRIP_SPAN / 2 - 0.008)
GRIP_DZ = 0.005  # the TCP above the grip's centre line (pad tips 8 mm below it)
DRIVE_YAW = {"left": math.pi / 2, "right": math.pi / 2}  # hand straight down, pads closing along x
APPROACH = 0.008  # the bit's tip this far short of the screw's head while it is lined up
PUSH = 23.0  # N: the push on the bit the demo aims for while driving (the screw turns from SR.F_PUSH, 20 N)
JOINT_K = np.array([SHOULDER["stiffness"]] * 3 + [WRIST["stiffness"]] * 3)  # the arm's joint drives, N m/rad
DRIVE_S = 4.5  # time to follow a screw home (the screw takes ~1.5 s at the driver's speed, pushed)
READ_GAIN = 0.6  # each read of the driver applies this share of the error it finds,
READ_TURN = math.radians(5.0)  # and turns the hand at most this much (near the edge of the arms' reach, a full
# correction overshot: the right arm ended 11 mm and 9 degrees off after three reads)
SCREW_ORDER = {"left": (2,), "right": (1, 0)}  # the marks whose screws each arm drives, in order


def _face(slot: int, seated: bool = False) -> np.ndarray:
    """The centre of a screw head's face, as started or seated."""
    x = (SCREW_SEATED_X if seated else SCREW_HEAD_X) - HEAD_T
    return np.array((x, SLOT_Y[slot], SCREW_Z))


def _bit_turn(R_driver):
    """The smallest turn that lines the bit (the driver's x axis) up with the screws' axis (+x), at most MAX_TURN;
    the driver's roll about the bit does not matter."""
    a, b = R_driver[:, 0], np.array((1.0, 0.0, 0.0))
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    angle = min(math.atan2(s, float(a @ b)), MAX_TURN)
    return axis_angle_matrix(v / s, angle)


PAD_TWIST = 0.6  # N m: the twist about the vertical the plan has the pads pass on to the grip (they hold ~0.9)


def _jacobian(q, base, offset_local):
    """The 6x6 Jacobian (linear velocity of a point fixed in the hand, ``offset_local`` in the TCP frame; angular
    velocity), numerically."""
    def pose(qq):
        p, R = ik.tcp_pose(qq, base)
        return p + R @ offset_local, R

    p0, R0 = pose(q)
    J = np.zeros((6, 6))
    for k in range(6):
        dq = np.zeros(6)
        dq[k] = 1e-6
        p1, R1 = pose(q + dq)
        J[:3, k] = (p1 - p0) / 1e-6
        J[3:, k] = ik._rot_error(R1, R0) / 1e-6
    return J


def _pressing(arm: str, hand_pos, hand_R, tip, force: float):
    """The hand pose to command so that, with the bit's tip held at ``tip`` by a screw's head, the hand pushes the
    grip with ``force`` N along the screw's axis and settles at (hand_pos, hand_R). The grip is 71 mm off the bit's
    axis, so the push twists the driver about the tip (1.8 N m at 25 N), more than the pads can pass on: the plan has
    the pads take PAD_TWIST of it and the hand push the grip sideways for the rest, which the tip's friction on the
    head takes (about 0.45 N per N of push, within the ~0.9 friction there). Each joint drive is a spring, so the
    command leads the pose by K^-1 J^T W in the joints (J at the grip), carried back to a hand pose through the arm's
    kinematics; and the driver gives F / (2 k) in the pads, so the hand goes that much further."""
    base = np.asarray(station.BASES[arm], float)
    q, _, _ = ik.solve(hand_pos, ik.matrix_to_quat_xyzw(hand_R), None, base_pos=base, iters=200)
    p0, R0 = ik.tcp_pose(q, base)
    grip = np.asarray(tip, float) + np.asarray(driver_grip(arm))  # the driver square, as the plan holds it
    r = np.asarray(tip, float) - grip
    full = -r[1] * force  # the push's twist about +z, by the pads on the grip up to PAD_TWIST
    twist = math.copysign(min(PAD_TWIST, abs(full)), full)
    side = (r[1] * force + twist) / r[0]  # by the hand on the grip, along y
    wrench = np.array((force, side, 0.0, 0.0, 0.0, twist))
    q_cmd = np.clip(q + (_jacobian(q, base, R0.T @ (grip - p0)).T @ wrench) / JOINT_K, ik.LOWER, ik.UPPER)
    p1, R1 = ik.tcp_pose(q_cmd, base)
    return p1 + SR.AXIS * force / (2 * FINGER_K), R1


def _drive_waypoints(arm: str, slots, yaw: float) -> list[Waypoint]:
    fix = {"dp": np.zeros(3), "R": np.eye(3)}  # the reads' correction to the planned hand pose
    grip_local = np.asarray(driver_grip(arm)) + np.array((0.0, 0.0, GRIP_DZ))
    up = np.array((0.0, 0.0, 0.06))
    rest_tip = np.asarray(DRIVER_STARTS[arm])
    driver = DRIVER_NAMES[arm]
    hand_rot = _original_grip_rotation(yaw)

    def plan(tip):
        return np.asarray(tip, float) + grip_local  # the driver square, as it lies on its rest

    def at(tip):
        return lambda: (plan(tip) + fix["dp"], fix["R"] @ hand_rot)

    def pressing(tip, force=PUSH):
        """The hand, read corrections in, pushing the bit's tip at ``tip`` (a head's face) with ``force``."""
        def where():
            hand, R = at(tip)()
            return _pressing(arm, hand, R, tip, force)

        return where

    def read(tip, label, sideways_only=False):
        def where():
            hand = plan(tip) + fix["dp"]
            st = _stud_now(driver)
            if st is not None:
                full = _bit_turn(st.rotation())
                angle = _turn_angle(full)
                share = min(READ_GAIN, READ_TURN / angle) if angle > 1e-9 else READ_GAIN
                turn = ik.slerp_matrix(np.eye(3), full, share)
                tip_now = np.asarray(st.origin, float)
                moved = np.asarray(tip, float) + turn @ (hand - tip_now)  # turn about the bit's tip, then move it
                wanted = moved - plan(tip)
                if sideways_only:  # while driving: keep the bit on the screw's axis, leave the push alone
                    wanted[0] = fix["dp"][0]
                fix["dp"] = np.clip(fix["dp"] + READ_GAIN * (wanted - fix["dp"]), -MAX_SHIFT, MAX_SHIFT)
                fix["R"] = turn @ fix["R"]
                err = (tip_now - np.asarray(tip)) * 1000
                print(f"DRIVER_READ {label}: bit tip off by x {err[0]:+.1f} y {err[1]:+.1f} z {err[2]:+.1f} mm, "
                      f"bit turned {math.degrees(_turn_angle(turn)):.1f} deg", flush=True)
            return at(tip)()

        return where

    wps = [
        Waypoint(2.0, tuple(plan(rest_tip) + up), FINGER_OPEN, "over the impact driver", yaw),
        Waypoint(1.5, tuple(plan(rest_tip)), FINGER_OPEN, "down to its grip", yaw),
        Waypoint(0.8, tuple(plan(rest_tip)), FINGER_HOLD, "hold the grip (trigger not pulled)", yaw),
        Waypoint(1.2, tuple(plan(rest_tip) + up), FINGER_HOLD, "lift it", yaw),
    ]
    for slot in slots:
        start, seated = _face(slot), _face(slot, seated=True)
        lined_up = start - np.array((APPROACH, 0.0, 0.0))
        name = f"screw {slot + 1}"
        wps += [
            # Straight across, low, as in the earlier full-score demo. The right driver brushes its arm on the way
            # to the right screw; it stays in the hand because the pad on the trigger seats on a broad face.
            Lazy(2.5, at(lined_up + np.array((0.0, 0.0, 0.05))), FINGER_HOLD, f"carry the driver to {name}"),
            Lazy(1.2, at(lined_up), FINGER_HOLD, "bit in line with the screw"),
            Lazy(0.6, at(lined_up), FINGER_HOLD, "settle"),
            Lazy(0.8, read(lined_up, f"{name} lined up"), FINGER_HOLD, "line the bit up"),
            Lazy(0.6, at(lined_up), FINGER_HOLD, "settle"),
            Lazy(0.8, read(lined_up, f"{name} lined up, again"), FINGER_HOLD, "line it up again"),
            Lazy(0.6, at(lined_up), FINGER_HOLD, "settle"),
            Lazy(0.8, read(lined_up, f"{name} lined up, third"), FINGER_HOLD, "line it up a third time"),
            Lazy(0.6, at(lined_up), FINGER_HOLD, "settle"),
            Lazy(0.8, read(lined_up, f"{name} lined up, fourth"), FINGER_HOLD, "line it up a fourth time"),
            Lazy(1.0, at(start - np.array((0.004, 0.0, 0.0))), FINGER_HOLD, "bit just short of the head"),
            Lazy(0.5, at(start - np.array((0.004, 0.0, 0.0))), FINGER_HOLD, "settle"),
            Lazy(0.6, read(start - np.array((0.004, 0.0, 0.0)), f"{name} at the head"), FINGER_HOLD,
                 "line it up at the head"),
            Lazy(0.5, at(start - np.array((0.004, 0.0, 0.0))), FINGER_HOLD, "settle"),
            Lazy(0.6, read(start - np.array((0.004, 0.0, 0.0)), f"{name} at the head, again"), FINGER_HOLD,
                 "line it up at the head again"),
            Lazy(0.8, at(start + np.array((0.001, 0.0, 0.0))), FINGER_HOLD, "bit onto the screw's head"),
            Lazy(0.4, at(start + np.array((0.001, 0.0, 0.0))), FINGER_HOLD, "settle"),
            Lazy(1.5, pressing(start), FINGER_HOLD, "press the bit into the head"),
            Lazy(0.6, pressing(start), FINGER_SQUEEZE, "pull the trigger"),
            # One steady move from the head to seated, pressing all the way: the screw turns only while the bit is
            # pressed in, so it goes in as fast as the hand follows. (A read part-way, which set the hand back to
            # the plan without the press, let a screw that had run ahead of the plan lose its bit 6 mm short.)
            Lazy(DRIVE_S, pressing(seated), FINGER_SQUEEZE, "follow the screw home, pressing"),
            Lazy(0.6, pressing(seated), FINGER_SQUEEZE, "hold"),
            # Ease the push off before relaxing the grip: let go of the trigger while still pushing ~23 N, the
            # pads on the grip were left almost unloaded and the right driver fell out of the hand.
            Lazy(1.0, at(seated - np.array((0.002, 0.0, 0.0))), FINGER_SQUEEZE, "ease off"),
            Lazy(0.4, at(seated - np.array((0.002, 0.0, 0.0))), FINGER_HOLD, "let go of the trigger"),
            Lazy(1.0, at(seated - np.array((0.03, 0.0, 0.0))), FINGER_HOLD, "back the bit out"),
            Lazy(1.0, at(seated - np.array((0.03, 0.0, -0.05))), FINGER_HOLD, "up"),
        ]
    # The arm keeps the driver, lifted clear of the work (set back on its rest, it was dropped beside it).
    return wps


def demo():
    """The left arm seats the left stud, the right arm the centre stud and then the right one; then each arm takes
    up its impact driver, drives the screws at its studs and sets the driver back on its rest."""
    plans = {"left": [], "right": []}
    by_name = {s.name: s for s in STUDS}
    for name, (arm, slot) in ASSIGNMENT.items():
        plans[arm].extend(_seat_waypoints(by_name[name], slot))
    for arm in ("left", "right"):
        plans[arm].extend(_drive_waypoints(arm, SCREW_ORDER[arm], DRIVE_YAW[arm]))
    return plans
