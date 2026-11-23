"""Mount a spring-loaded towel bar between two wall brackets.

A small wall section stands upright on the table with two bracket housings protruding from its face,
their channels facing inward.  The towel bar has a spring-loaded post on its left end that retracts
when the assembly is pushed against the left bracket, shortening it so the right end clears the right
bracket.  Slide the assembly right so the right end enters the right channel; the spring extends the
post into the left channel, clamping the bar between both brackets.

Domain: Impact on jobs — general maintenance and repair workers (O*NET 49-9071, 1.5 M; task
statement: "Repair or replace defective equipment, machines, or structures, such as ... plumbing
fixtures") and janitors and cleaners (37-2011, 2.3 M; task: "Install equipment") regularly replace
or refit spring-loaded towel bars, grab bars and shelf rails in hotels, hospitals, schools and
office buildings.

How it is modelled (see jointkit.py): the wall plate and two bracket housings are a single kinematic
body.  The towel bar is a free dynamic body.  The spring post is a second dynamic body on a slider
joint attached to the bar, with a spring drive pushing it outward (+y).  Contact forces between the
bar end / spring post and the bracket channel walls provide the guidance and clamping.

Scoring: 0.25 spring compressed while bar aligned with brackets; + 0.25 right end in its channel;
+ 0.25 both ends in channels (bar spanning both brackets); + 0.25 hands clear and bar stable 1 s.

Run it:  inspect-robots run --embodiment yam_arena -E task=mount_towel_bar.py -E object_jitter=0 \\
           --instruction "Mount the towel bar between the two wall brackets." \\
           --scorer task_score --max-steps 2400 --no-prompt --store-frames
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from jointkit import (  # noqa: E402
    Body, BoxPart, CylPart, as_numpy, on_reset, on_step, pose, rotation, set_drive, slider,
)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- geometry ---------------------------------------------------------------------------------------------------------

WALL_X = 0.27
WALL_W = 0.35
WALL_H = 0.25
WALL_D = 0.020
WALL_FRONT = WALL_X - WALL_D / 2

BRACKET_Z = TOP + 0.10
BRACKET_Y_L = 0.120
BRACKET_Y_R = -0.120

CH_INNER = 0.050
CH_DEPTH = 0.055
CH_WALL = 0.003
PROTRUSION = CH_INNER + 2 * CH_WALL

BAR_LEN = 0.220
BAR_W = 0.018
BAR_MASS = 0.25
BAR_START_X = 0.15
BAR_START_Z = TOP + BAR_W / 2 + 0.001

POST_BODY = 0.015
POST_W = 0.016
POST_MASS = 0.04
SLIDE_MAX = 0.040
SPRING_K = 250.0
SPRING_D = 30.0

ITERATIONS = (64, 8)

# -- colours ----------------------------------------------------------------------------------------------------------

WALL_C = (0.85, 0.83, 0.80)
BRACKET_C = (0.80, 0.82, 0.84)
CHROME = (0.78, 0.80, 0.82)
POST_C = (0.70, 0.72, 0.74)

# -- scoring ----------------------------------------------------------------------------------------------------------

SEAT_DEPTH = 0.006
LEVEL_DEG = 5.0
CLEAR_DIST = 0.04
HOLD_S = 1.0
STILL_V = 0.003
STILL_W = math.radians(3.0)

SCORE_OVER = "end"


# -- build objects ----------------------------------------------------------------------------------------------------

def _bracket_parts(y_open: float, toward_neg_y: bool, bz: float) -> list[BoxPart]:
    """Five-wall channel protruding from the wall face.  ``y_open`` is the channel opening
    in body-local y; ``toward_neg_y`` True means the channel opens toward -y."""
    sign = 1.0 if toward_neg_y else -1.0
    bx = -(WALL_D / 2 + PROTRUSION / 2)
    cy = y_open + sign * CH_DEPTH / 2
    hi = CH_INNER / 2
    hcw = CH_WALL / 2
    parts = []
    parts.append(BoxPart((bx, cy, bz - hi - hcw),
                         (PROTRUSION, CH_DEPTH, CH_WALL), BRACKET_C))
    parts.append(BoxPart((bx, cy, bz + hi + hcw),
                         (PROTRUSION, CH_DEPTH, CH_WALL), BRACKET_C))
    parts.append(BoxPart((bx - hi - hcw, cy, bz),
                         (CH_WALL, CH_DEPTH, CH_INNER), BRACKET_C))
    parts.append(BoxPart((bx + hi + hcw, cy, bz),
                         (CH_WALL, CH_DEPTH, CH_INNER), BRACKET_C))
    back_y = y_open + sign * (CH_DEPTH + hcw)
    parts.append(BoxPart((bx, back_y, bz),
                         (PROTRUSION, CH_WALL, PROTRUSION), BRACKET_C))
    return parts


def _wall_plate() -> Body:
    body_z = TOP + WALL_H / 2
    bz = BRACKET_Z - body_z
    parts = [BoxPart((0.0, 0.0, 0.0), (WALL_D, WALL_W, WALL_H), WALL_C, collide=False)]
    back_thick = 0.020
    for side_parts in (_bracket_parts(BRACKET_Y_L, toward_neg_y=True, bz=bz),
                       _bracket_parts(BRACKET_Y_R, toward_neg_y=False, bz=bz)):
        for i, bp in enumerate(side_parts):
            if i == 4:
                sx, sy, sz = bp.size
                parts.append(BoxPart(bp.centre, (sx, back_thick, sz), bp.color, collide=True))
            elif i == 0:
                parts.append(BoxPart(bp.centre, bp.size, bp.color, collide=True))
            else:
                parts.append(BoxPart(bp.centre, bp.size, bp.color, collide=False))
    return Body.build("wall_plate", pos=(WALL_X, 0.0, body_z),
                      mass=10.0, parts=parts, kinematic=True, color=WALL_C)


def _towel_bar() -> Body:
    parts = [BoxPart((0.0, 0.0, 0.0), (BAR_W, BAR_LEN, BAR_W), CHROME)]
    return Body.build("bar", pos=(BAR_START_X, 0.0, BAR_START_Z), mass=BAR_MASS,
                      parts=parts, color=CHROME, friction=(0.5, 0.4), iterations=ITERATIONS)


def _spring_post() -> Body:
    parts = [BoxPart((0.0, 0.0, 0.0), (POST_W, POST_BODY, POST_W), POST_C)]
    return Body.build("post", pos=(BAR_START_X, BAR_LEN / 2, BAR_START_Z), mass=POST_MASS,
                      parts=parts, color=POST_C, friction=(0.5, 0.4),
                      ignore=("bar",), iterations=ITERATIONS)


OBJECTS = [_wall_plate(), _towel_bar(), _spring_post()]

_ANCHOR = (BAR_START_X, BAR_LEN / 2, BAR_START_Z)
slider("post_spring", "bar", "post", anchor=_ANCHOR, axis="Y",
       limits=(0.0, SLIDE_MAX), stiffness=SPRING_K, damping=SPRING_D,
       target=SLIDE_MAX, max_force=50.0)

INSTRUCTION = "Mount the towel bar between the two wall brackets."

# -- channel geometry in world coordinates ----------------------------------------------------------------------------

_BODY_Z = TOP + WALL_H / 2
_BX_W = WALL_X - (WALL_D / 2 + PROTRUSION / 2)

L_CH_OPEN = np.array([_BX_W, BRACKET_Y_L, BRACKET_Z])
L_CH_BACK = np.array([_BX_W, BRACKET_Y_L + CH_DEPTH, BRACKET_Z])
R_CH_OPEN = np.array([_BX_W, BRACKET_Y_R, BRACKET_Z])
R_CH_BACK = np.array([_BX_W, BRACKET_Y_R - CH_DEPTH, BRACKET_Z])


def _in_channel(point: np.ndarray, ch_open: np.ndarray, ch_back: np.ndarray) -> float:
    dx = abs(float(point[0] - ch_open[0]))
    dz = abs(float(point[2] - ch_open[2]))
    if dx > CH_INNER / 2 or dz > CH_INNER / 2:
        return 0.0
    if ch_back[1] > ch_open[1]:
        depth = float(point[1] - ch_open[1])
        return max(depth, 0.0)
    else:
        depth = float(ch_open[1] - point[1])
        return max(depth, 0.0)


# -- runtime state ----------------------------------------------------------------------------------------------------

def _fresh() -> dict:
    return {"compressed_seen": False, "right_seated": False,
            "both_seated": False, "stable_steps": 0, "tcps": None}


_EP = _fresh()


@on_reset
def _new_episode() -> None:
    _EP.clear()
    _EP.update(_fresh())
    _DBG["steps"] = 0


def _tcps(env) -> list[np.ndarray]:
    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names)
                 if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        out.append(g[:3] - origin + rotation(g[3:7]) @ np.array(TCP_OFFSET))
    return out


def _post_tip(env) -> np.ndarray:
    post_pos, post_q = pose(env, "post")
    r = rotation(post_q)
    return post_pos + r @ np.array([0.0, POST_BODY / 2, 0.0])


def _bar_right_end(env) -> np.ndarray:
    bar_pos, bar_q = pose(env, "bar")
    r = rotation(bar_q)
    return bar_pos - r @ np.array([0.0, BAR_LEN / 2, 0.0])


def _bar_level(env) -> float:
    _, bar_q = pose(env, "bar")
    r = rotation(bar_q)
    ax = r @ np.array([0.0, 1.0, 0.0])
    return math.degrees(math.asin(min(abs(float(ax[2])), 1.0)))


def _slider_ext(env) -> float:
    bar_pos, bar_q = pose(env, "bar")
    post_pos, _ = pose(env, "post")
    r = rotation(bar_q)
    local = r.T @ (post_pos - bar_pos)
    return float(local[1]) - BAR_LEN / 2


@on_step
def _follow(env) -> None:
    ext = _slider_ext(env)
    right_end = _bar_right_end(env)
    ptip = _post_tip(env)
    level = _bar_level(env)

    near_height = (abs(float(right_end[2]) - BRACKET_Z) < 0.04
                   and abs(float(ptip[2]) - BRACKET_Z) < 0.04)
    near_wall = (abs(float(right_end[0]) - _BX_W) < 0.04
                 and abs(float(ptip[0]) - _BX_W) < 0.04)
    compressed = ext < SLIDE_MAX * 0.5

    if compressed and near_height and near_wall:
        _EP["compressed_seen"] = True

    r_depth = _in_channel(right_end, R_CH_OPEN, R_CH_BACK)
    if r_depth >= SEAT_DEPTH:
        _EP["right_seated"] = True

    l_depth = _in_channel(ptip, L_CH_OPEN, L_CH_BACK)
    if l_depth >= SEAT_DEPTH and r_depth >= SEAT_DEPTH and level < LEVEL_DEG:
        _EP["both_seated"] = True

    _EP["tcps"] = _tcps(env)

    vel = as_numpy(env.unwrapped.scene["bar"].data.root_com_vel_w
                   if hasattr(env.unwrapped.scene["bar"].data, "root_com_vel_w")
                   else env.unwrapped.scene["bar"].data.root_vel_w)[0]
    if _EP["both_seated"] and float(np.linalg.norm(vel[:3])) < STILL_V and float(np.linalg.norm(vel[3:])) < STILL_W:
        _EP["stable_steps"] += 1
    elif not _EP["both_seated"]:
        _EP["stable_steps"] = 0

    if _DBG["on"] and _DBG["steps"] % 20 == 0:
        t = _DBG["steps"] / 20
        bar_pos, _ = pose(env, "bar")
        post_pos, _ = pose(env, "post")
        tcps = _tcps(env)
        print(f"[bar] t={t:5.1f}s ext={ext*1000:5.1f}mm lev={level:4.1f} "
              f"Rd={r_depth*1000:5.1f} Ld={l_depth*1000:5.1f} "
              f"bar=({bar_pos[0]:.3f},{bar_pos[1]:.3f},{bar_pos[2]:.3f}) "
              f"post=({post_pos[0]:.3f},{post_pos[1]:.3f},{post_pos[2]:.3f}) "
              f"Ltcp=({tcps[0][0]:.3f},{tcps[0][1]:.3f},{tcps[0][2]:.3f}) "
              f"Rtcp=({tcps[1][0]:.3f},{tcps[1][1]:.3f},{tcps[1][2]:.3f})",
              file=sys.stderr, flush=True)
    _DBG["steps"] += 1


_DBG = {"on": bool(os.environ.get("BAR_DEBUG")), "steps": 0}


# -- scoring ----------------------------------------------------------------------------------------------------------


def score(objects) -> float:
    for name in ("bar", "post", "wall_plate"):
        o = objects.get(name)
        if o is None or not all(np.all(np.isfinite(v)) for v in (o.pos, o.quat_xyzw)):
            return 0.0

    pts = 0.0
    if _EP["compressed_seen"]:
        pts += 0.25
    if _EP["right_seated"]:
        pts += 0.25
    if _EP["both_seated"]:
        pts += 0.25
    if _EP["both_seated"] and _EP["stable_steps"] >= int(HOLD_S * 20):
        tcps = _EP.get("tcps")
        if tcps is not None:
            bp = objects["bar"].pos
            pp = objects["post"].pos
            clear = all(
                float(np.linalg.norm(tcp - bp)) >= CLEAR_DIST
                and float(np.linalg.norm(tcp - pp)) >= CLEAR_DIST
                for tcp in tcps)
            if clear:
                pts += 0.25
    return pts


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- scripted demo ----------------------------------------------------------------------------------------------------

HOVER = 0.06
OUT_L = (0.14, 0.22, TOP + 0.20)
OUT_R = (0.14, -0.22, TOP + 0.20)

MOUNT_X = _BX_W
MOUNT_Z = BRACKET_Z
GRIP_YAW = math.pi / 2


def demo():
    W = Waypoint

    grip_ly = 0.10
    grip_ry = -0.10
    pick_z = BAR_START_Z
    hover_z = TOP + 0.10
    mx = MOUNT_X
    mz = MOUNT_Z

    left = [
        W(5.0, (BAR_START_X, grip_ly, hover_z), FINGER_OPEN, "approach left", GRIP_YAW),
        W(2.0, (BAR_START_X, grip_ly, hover_z), FINGER_OPEN, "settle", GRIP_YAW),
        W(3.0, (BAR_START_X, grip_ly, pick_z), FINGER_OPEN, "descend", GRIP_YAW),
        W(1.0, (BAR_START_X, grip_ly, pick_z), FINGER_CLOSED, "grip", GRIP_YAW),
    ]
    right = [
        W(5.0, (BAR_START_X, grip_ry, hover_z), FINGER_OPEN, "approach right", GRIP_YAW),
        W(2.0, (BAR_START_X, grip_ry, hover_z), FINGER_OPEN, "settle", GRIP_YAW),
        W(3.0, (BAR_START_X, grip_ry, pick_z), FINGER_OPEN, "descend", GRIP_YAW),
        W(1.0, (BAR_START_X, grip_ry, pick_z), FINGER_CLOSED, "grip", GRIP_YAW),
    ]

    left += [
        W(5.0, (BAR_START_X, grip_ly, mz), FINGER_CLOSED, "lift", GRIP_YAW),
        W(5.0, (mx, grip_ly, mz), FINGER_CLOSED, "forward", GRIP_YAW),
        W(3.0, (mx, grip_ly, mz), FINGER_CLOSED, "settle at wall", GRIP_YAW),
    ]
    right += [
        W(5.0, (BAR_START_X, grip_ry, mz), FINGER_CLOSED, "lift", GRIP_YAW),
        W(5.0, (mx, grip_ry, mz), FINGER_CLOSED, "forward", GRIP_YAW),
        W(3.0, (mx, grip_ry, mz), FINGER_CLOSED, "settle at wall", GRIP_YAW),
    ]

    push_left = 0.060
    left += [W(8.0, (mx, grip_ly + push_left, mz), FINGER_CLOSED,
               "push left, compress spring", GRIP_YAW)]
    right += [W(8.0, (mx, grip_ry + push_left, mz), FINGER_CLOSED,
                "push left together", GRIP_YAW)]

    slide_right = 0.084
    left += [
        W(8.0, (mx, grip_ly + push_left - slide_right, mz), FINGER_CLOSED,
            "slide right", GRIP_YAW),
        W(5.0, (mx, grip_ly + push_left - slide_right, mz), FINGER_CLOSED,
            "hold steady", GRIP_YAW),
    ]
    right += [
        W(8.0, (mx, grip_ry + push_left - slide_right, mz), FINGER_CLOSED,
            "slide right", GRIP_YAW),
        W(5.0, (mx, grip_ry + push_left - slide_right, mz), FINGER_CLOSED,
            "hold steady", GRIP_YAW),
    ]

    ly_final = grip_ly + push_left - slide_right
    ry_final = grip_ry + push_left - slide_right
    left += [
        W(1.0, (mx, ly_final, mz), FINGER_OPEN, "release", GRIP_YAW),
        W(3.0, OUT_L, FINGER_OPEN, "clear", GRIP_YAW),
    ]
    right += [
        W(1.0, (mx, ry_final, mz), FINGER_OPEN, "release", GRIP_YAW),
        W(3.0, OUT_R, FINGER_OPEN, "clear", GRIP_YAW),
    ]

    end = max(sum(w.duration for w in left), sum(w.duration for w in right)) + 5.0
    for plan in (left, right):
        plan.append(W(end - sum(w.duration for w in plan), plan[-1].pos, FINGER_OPEN, "wait", GRIP_YAW))
    return {"left": left, "right": right}


def _read_bar_yaw(ctx) -> tuple[float, float, float]:
    """Return (bar_x, bar_y, yaw_rad) from live sim state."""
    env = ctx.emb._env
    bar_pos, bar_q = pose(env, "bar")
    post_pos, _ = pose(env, "post")
    r = rotation(bar_q)
    local = r.T @ (post_pos - bar_pos)
    bar_yaw = math.atan2(-float(local[0]), float(local[1]))
    return float(bar_pos[0]), float(bar_pos[1]), bar_yaw


def demo_script(ctx):
    """Closed-loop demo for cabledemo.py — staged warm-up ensures both arms grip."""
    yaw = GRIP_YAW
    grip_ly, grip_ry = 0.10, -0.10
    mx, mz = MOUNT_X, MOUNT_Z
    hz = TOP + 0.10

    # Wide hover near READY y — only 40 mm from rest position
    yield from ctx.both(
        ctx.left.move((BAR_START_X, 0.14, hz), yaw=yaw, duration=3.0),
        ctx.right.move((BAR_START_X, -0.14, hz), yaw=yaw, duration=3.0),
    )
    yield from ctx.both(ctx.left.wait(2.0), ctx.right.wait(2.0))

    # Slide inward to grip hover
    yield from ctx.both(
        ctx.left.move((BAR_START_X, grip_ly, hz), yaw=yaw, duration=3.0),
        ctx.right.move((BAR_START_X, grip_ry, hz), yaw=yaw, duration=3.0),
    )
    yield from ctx.both(ctx.left.wait(2.0), ctx.right.wait(2.0))

    # Convergence loop — keep commanding until both arms are within 30 mm of target y
    for _ in range(10):
        ly = float(ctx.left.tcp()[1])
        ry = float(ctx.right.tcp()[1])
        if abs(ly - grip_ly) < 0.03 and abs(ry - grip_ry) < 0.03:
            break
        yield from ctx.both(
            ctx.left.move((BAR_START_X, grip_ly, hz), yaw=yaw, duration=1.5),
            ctx.right.move((BAR_START_X, grip_ry, hz), yaw=yaw, duration=1.5),
        )

    # Descend to bar
    yield from ctx.both(
        ctx.left.move((BAR_START_X, grip_ly, BAR_START_Z), yaw=yaw, duration=2.0),
        ctx.right.move((BAR_START_X, grip_ry, BAR_START_Z), yaw=yaw, duration=2.0),
    )
    yield from ctx.both(ctx.left.wait(1.0), ctx.right.wait(1.0))
    yield from ctx.both(ctx.left.grip(FINGER_CLOSED), ctx.right.grip(FINGER_CLOSED))

    # Lift
    yield from ctx.both(
        ctx.left.move((BAR_START_X, grip_ly, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
        ctx.right.move((BAR_START_X, grip_ry, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
    )
    yield from ctx.both(ctx.left.wait(1.0), ctx.right.wait(1.0))

    # Staged forward approach
    mid_x = (BAR_START_X + mx) / 2
    yield from ctx.both(
        ctx.left.move((mid_x, grip_ly, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
        ctx.right.move((mid_x, grip_ry, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
    )
    yield from ctx.both(ctx.left.wait(1.0), ctx.right.wait(1.0))
    yield from ctx.both(
        ctx.left.move((mx, grip_ly, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
        ctx.right.move((mx, grip_ry, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
    )
    yield from ctx.both(ctx.left.wait(2.0), ctx.right.wait(2.0))

    # Yaw correction
    for _attempt in range(3):
        bx, _, byaw = _read_bar_yaw(ctx)
        if abs(byaw) < math.radians(5) and abs(bx - mx) < 0.020:
            break
        lx = float(ctx.left.tcp()[0])
        rx = float(ctx.right.tcp()[0])
        tgt_lx = lx + (mx - lx) * 0.7
        tgt_rx = rx + (mx - rx) * 0.7
        yield from ctx.both(
            ctx.left.move((tgt_lx, grip_ly, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
            ctx.right.move((tgt_rx, grip_ry, mz), yaw=yaw, finger=FINGER_CLOSED, duration=3.0),
        )
        yield from ctx.both(ctx.left.wait(1.0), ctx.right.wait(1.0))

    # Push left — compress spring
    pl = 0.060
    yield from ctx.both(
        ctx.left.move((mx, grip_ly + pl, mz), yaw=yaw, finger=FINGER_CLOSED, duration=8.0),
        ctx.right.move((mx, grip_ry + pl, mz), yaw=yaw, finger=FINGER_CLOSED, duration=8.0),
    )
    yield from ctx.both(ctx.left.wait(2.0), ctx.right.wait(2.0))

    # Slide right — targets bar center at y ≈ −0.024
    sr = 0.084
    yield from ctx.both(
        ctx.left.move((mx, grip_ly + pl - sr / 2, mz), yaw=yaw, finger=FINGER_CLOSED, duration=5.0),
        ctx.right.move((mx, grip_ry + pl - sr / 2, mz), yaw=yaw, finger=FINGER_CLOSED, duration=5.0),
    )
    yield from ctx.both(ctx.left.wait(1.0), ctx.right.wait(1.0))
    yield from ctx.both(
        ctx.left.move((mx, grip_ly + pl - sr, mz), yaw=yaw, finger=FINGER_CLOSED, duration=5.0),
        ctx.right.move((mx, grip_ry + pl - sr, mz), yaw=yaw, finger=FINGER_CLOSED, duration=5.0),
    )
    yield from ctx.both(ctx.left.wait(3.0), ctx.right.wait(3.0))

    # Release and clear
    yield from ctx.both(
        ctx.left.grip(FINGER_OPEN, duration=0.5),
        ctx.right.grip(FINGER_OPEN, duration=0.5),
    )
    yield from ctx.both(
        ctx.left.move(OUT_L, yaw=yaw, duration=2.0),
        ctx.right.move(OUT_R, yaw=yaw, duration=2.0),
    )
