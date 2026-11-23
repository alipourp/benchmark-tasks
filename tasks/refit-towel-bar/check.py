"""Offline scorer verification for the towel bar task.

Tests the scoring logic against six scripted variants without needing the
simulator.  Run:

    python check.py

Expected output: 6 passed, 0 failed.
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Scoring constants — must match mount_towel_bar.py
# ---------------------------------------------------------------------------

SEAT_DEPTH = 0.006
CLEAR_DIST = 0.04
HOLD_S = 1.0
HOLD_STEPS = int(HOLD_S * 20)


class _Obj:
    """Minimal stand-in for an object with pos and quat."""

    def __init__(self, pos, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
        self.pos = np.array(pos, dtype=float)
        self.quat_xyzw = np.array(quat_xyzw, dtype=float)


def _objects(bar_pos, post_pos):
    return {
        "bar": _Obj(bar_pos),
        "post": _Obj(post_pos),
        "wall_plate": _Obj((0.27, 0.0, 0.8725)),
    }


def _score(ep, objects):
    """Reproduce the score() logic from mount_towel_bar.py exactly."""
    for name in ("bar", "post", "wall_plate"):
        o = objects.get(name)
        if o is None or not all(
            np.all(np.isfinite(v)) for v in (o.pos, o.quat_xyzw)
        ):
            return 0.0
    pts = 0.0
    if ep["compressed_seen"]:
        pts += 0.25
    if ep["right_seated"]:
        pts += 0.25
    if ep["both_seated"]:
        pts += 0.25
    if ep["both_seated"] and ep["stable_steps"] >= HOLD_STEPS:
        tcps = ep.get("tcps")
        if tcps is not None:
            bp = objects["bar"].pos
            pp = objects["post"].pos
            clear = all(
                float(np.linalg.norm(tcp - bp)) >= CLEAR_DIST
                and float(np.linalg.norm(tcp - pp)) >= CLEAR_DIST
                for tcp in tcps
            )
            if clear:
                pts += 0.25
    return pts


# ---------------------------------------------------------------------------
# Scripted variants
# ---------------------------------------------------------------------------
#
#   Variant                                   Score   Why
#   ----------------------------------------  -----   ----------------------------
#   1  Bar untouched on table                 0.00    No flags set
#   2  Spring compressed near brackets only   0.25    compressed_seen
#   3  Right end seated, never compressed     0.25    right_seated only
#   4  Compressed + right end seated          0.50    compressed_seen + right_seated
#   5  Both seated, hands still on bar        0.75    + both_seated, hands too close
#   6  Full: both seated, stable, hands clear 1.00    all four conditions met

# TCP positions when arms are at rest (far from bar)
_REST_L = np.array([0.14, 0.22, 0.95])
_REST_R = np.array([0.14, -0.22, 0.95])

# TCP positions when hands are gripping the bar near the brackets
_CLOSE_L = np.array([0.232, 0.01, 0.8475])
_CLOSE_R = np.array([0.232, -0.01, 0.8475])

VARIANTS = [
    (
        "Bar untouched on table",
        0.00,
        dict(compressed_seen=False, right_seated=False, both_seated=False,
             stable_steps=0, tcps=[_REST_L, _REST_R]),
        (0.15, 0.0, 0.7585),
        (0.15, 0.11, 0.7585),
    ),
    (
        "Spring compressed near brackets, neither end seated",
        0.25,
        dict(compressed_seen=True, right_seated=False, both_seated=False,
             stable_steps=0, tcps=[_CLOSE_L, _CLOSE_R]),
        (0.232, 0.03, 0.8475),
        (0.232, 0.14, 0.8475),
    ),
    (
        "Right end seated, spring never compressed near brackets",
        0.25,
        dict(compressed_seen=False, right_seated=True, both_seated=False,
             stable_steps=0, tcps=[_CLOSE_L, _CLOSE_R]),
        (0.232, -0.02, 0.8475),
        (0.232, 0.10, 0.8475),
    ),
    (
        "Compressed + right end seated",
        0.50,
        dict(compressed_seen=True, right_seated=True, both_seated=False,
             stable_steps=0, tcps=[_CLOSE_L, _CLOSE_R]),
        (0.232, -0.02, 0.8475),
        (0.232, 0.10, 0.8475),
    ),
    (
        "Both ends seated, bar level, hands still on bar",
        0.75,
        dict(compressed_seen=True, right_seated=True, both_seated=True,
             stable_steps=HOLD_STEPS, tcps=[_CLOSE_L, _CLOSE_R]),
        (0.232, -0.02, 0.8475),
        (0.232, 0.13, 0.8475),
    ),
    (
        "Full score: both seated, stable, hands clear",
        1.00,
        dict(compressed_seen=True, right_seated=True, both_seated=True,
             stable_steps=HOLD_STEPS, tcps=[_REST_L, _REST_R]),
        (0.232, -0.02, 0.8475),
        (0.232, 0.13, 0.8475),
    ),
]


def main() -> int:
    passed = failed = 0
    for name, expected, ep, bar_pos, post_pos in VARIANTS:
        got = _score(ep, _objects(bar_pos, post_pos))
        ok = abs(got - expected) < 0.001
        tag = "PASS" if ok else "FAIL"
        extra = "" if ok else f"  (got {got:.2f})"
        print(f"  {tag}  {expected:.2f}  {name}{extra}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
