"""Checks of stud_layout_jig.py's scene, scoring, screw model and demo, without the simulator. Needs yam_arena (the
official stack):

    python -m pytest -q test_stud_layout_jig.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("stud_layout_jig.py"))
M = TASK.module
SR = M.SR
SLOT_OF = {name: slot for name, (_, slot) in M.ASSIGNMENT.items()}  # stud -> its mark in the demo


@pytest.fixture(autouse=True)
def _clean_screws():
    M.TEST_SCREWS = None
    yield
    M.TEST_SCREWS = None


def stud_state(x, y, yaw_deg=0.0, roll_deg=0.0, bottom=None, quat=None):
    """A stud with its centre at x, y, turned ``yaw_deg`` about the vertical and rolled ``roll_deg`` about its own
    length, its lowest point at ``bottom``. The world-aligned box comes from the turned half-extents, as the
    simulator reports it."""
    if quat is None:
        yaw, roll = math.radians(yaw_deg), math.radians(roll_deg)
        cr, sr, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(yaw / 2), math.sin(yaw / 2)
        quat = np.array([sr * cy, -sr * sy, cr * sy, cr * cy])  # Rz(yaw) * Rx(roll), xyzw
    quat = np.asarray(quat, dtype=float)
    probe = ObjectState(np.zeros(3), quat, np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3))
    half = np.abs(M._rotation(probe)) @ M.STUD_HALF
    centre = np.array([x, y, (M.BASE_TOP if bottom is None else bottom) + half[2]])
    return ObjectState(centre, quat, np.zeros(3), np.zeros(3), centre - half, centre + half)


def on_marks(slot, gap=0.0, off_mark=0.0, **kw):
    """A stud on edge on its marks, its end ``gap`` from the bottom plate (0: tight against it)."""
    return stud_state(M.SEATED_X + gap, M.SLOT_Y[slot] + off_mark, **kw)


def all_on_marks(**override):
    objects = {name: on_marks(slot) for name, slot in SLOT_OF.items()}
    objects.update(override)
    return objects


def screws(depth=None, caught=True, **per_slot):
    """The screws' state by mark: each catches the stud on its mark (when ``caught``) and is driven ``depth`` in
    (default: home). ``per_slot`` overrides one mark: slot_0=(stud or None, depth)."""
    depth = M.SCREW_PROUD if depth is None else depth
    by_slot = {slot: name for name, slot in SLOT_OF.items()}
    out = [(by_slot[i] if caught else None, depth) for i in range(len(M.SLOT_Y))]
    for key, value in per_slot.items():
        out[int(key.split("_")[1])] = value
    return out


def at_start():
    return {s.name: stud_state(s.pos[0], s.pos[1], math.degrees(s.yaw), bottom=s.pos[2] - M.STUD_DEPTH / 2)
            for s in M.STUDS}


# -- scene -------------------------------------------------------------------------------------


def test_real_lumber_sizes_spacing_and_mass():
    assert M.STUD_THICK == pytest.approx(0.0381)
    assert M.STUD_DEPTH == pytest.approx(0.0889)
    assert M.STUD_LENGTH == pytest.approx(12 * 0.0254)
    assert 0.40 < M.STUD_MASS < 0.50
    assert M.SLOT_Y[1] - M.SLOT_Y[0] == pytest.approx(16 * 0.0254)
    assert M.SLOT_Y[2] - M.SLOT_Y[1] == pytest.approx(16 * 0.0254)
    assert M.PLATE_LEN == pytest.approx(33.5 * 0.0254)


def test_plates_stand_a_stud_length_plus_six_millimetres_apart():
    assert M.TOP_PLATE_FACE_X - M.BOTTOM_PLATE_INNER_X == pytest.approx(M.STUD_LENGTH + 0.006)


def test_free_objects_are_the_studs_screws_and_drivers():
    assert len(set(TASK.names)) == len(TASK.names)
    expected = {s.name for s in M.STUDS} | {s.name for s in M.SCREWS} | set(M.DRIVER_NAMES.values())
    assert set(TASK.free_objects) == expected


def test_screws_start_with_their_tips_flush_and_heads_proud():
    for screw in M.SCREWS:
        assert screw.pos[0] + M.SCREW_LEN == pytest.approx(M.BOTTOM_PLATE_INNER_X)
        assert M.BOTTOM_PLATE_OUTER_X - screw.pos[0] == pytest.approx(1.5 * 0.0254)


def test_pilot_holes_clear_the_shanks():
    assert M.HOLE_HALF > M.SHANK_R


def test_driver_is_held_at_its_balance_point():
    driver = M.make_driver((0.0, 0.0, 0.0))
    assert driver.volume_centroid()[1] == pytest.approx(M.DRIVER_GRIP[1])


@pytest.mark.parametrize("slot", range(3))
def test_rounded_caps_keep_the_slot_width(slot):
    """Each cap sphere meets its cleat's slot-side face at the top of the square part, never inside the slot."""
    y = M.SLOT_Y[slot]
    for obj in M.OBJECTS:
        if obj.name.startswith(f"cleat_{slot + 1}") and "_round_" in obj.name:
            assert abs(obj.pos[1] - y) - obj.radius == pytest.approx(M.SLOT_W / 2, abs=1e-9)
            assert obj.pos[2] + obj.radius == pytest.approx(M.BASE_TOP + M.CLEAT_H)


# -- scoring -----------------------------------------------------------------------------------


def test_full_score_when_every_stud_is_on_its_marks_and_fastened():
    M.TEST_SCREWS = screws()
    assert M.score(all_on_marks()) == pytest.approx(1.0)
    assert M.success(all_on_marks())


def test_nothing_done_scores_zero():
    M.TEST_SCREWS = screws(depth=0.0, caught=False)
    assert M.score(at_start()) == 0.0
    assert not M.success(at_start())


def test_studs_on_their_marks_but_not_fastened():
    M.TEST_SCREWS = screws(depth=0.0, caught=False)
    assert M.score(all_on_marks()) == pytest.approx(3 * M.PLACED_CREDIT)
    assert not M.success(all_on_marks())


def test_screws_caught_but_not_driven_home():
    M.TEST_SCREWS = screws(depth=0.020)
    assert M.score(all_on_marks()) == pytest.approx(3 * (M.PLACED_CREDIT + M.CAUGHT_CREDIT))


@pytest.mark.parametrize("short,home", [(0.0, True), (0.0009, True), (0.0011, False), (0.005, False)])
def test_a_screw_is_home_within_one_millimetre_of_flush(short, home):
    M.TEST_SCREWS = screws(slot_0=(next(n for n, s in SLOT_OF.items() if s == 0), M.SCREW_PROUD - short))
    expected = 1.0 if home else 1.0 - M.FASTENED_CREDIT
    assert M.score(all_on_marks()) == pytest.approx(expected)


def test_a_screw_home_with_the_stud_not_tight_does_not_fasten_it():
    name = next(n for n, s in SLOT_OF.items() if s == 0)
    M.TEST_SCREWS = screws()
    assert M.score(all_on_marks(**{name: on_marks(0, gap=0.002)})) == pytest.approx(1.0 - M.FASTENED_CREDIT)


def test_a_screw_driven_into_the_gap_holds_nothing():
    M.TEST_SCREWS = screws(slot_0=(None, M.SCREW_PROUD))
    assert M.score(all_on_marks()) == pytest.approx(1.0 - M.CAUGHT_CREDIT - M.FASTENED_CREDIT)


@pytest.mark.parametrize(
    "label,stud",
    [
        ("5 mm off its marks", lambda: on_marks(0, off_mark=0.005)),
        ("laid flat on the cleats", lambda: on_marks(0, roll_deg=90, bottom=M.BASE_TOP + M.CLEAT_H)),
        ("resting on the cleat tops", lambda: on_marks(0, bottom=M.BASE_TOP + M.CLEAT_H)),
        ("tilted 5 degrees", lambda: on_marks(0, roll_deg=5)),
        ("turned 5 degrees", lambda: on_marks(0, yaw_deg=5)),
        ("on the table, not the jig", lambda: on_marks(0, bottom=M.TOP)),
        ("over the bottom plate", lambda: on_marks(0, gap=-0.02)),
    ],
)
def test_a_stud_not_on_its_marks_earns_nothing(label, stud):
    name = next(n for n, s in SLOT_OF.items() if s == 0)
    M.TEST_SCREWS = screws()
    assert M.score(all_on_marks(**{name: stud()})) == pytest.approx(2 / 3), label


def test_a_stud_within_four_millimetres_of_its_marks_counts():
    name = next(n for n, s in SLOT_OF.items() if s == 0)
    M.TEST_SCREWS = screws()
    assert M.score(all_on_marks(**{name: on_marks(0, off_mark=0.0035)})) == pytest.approx(1.0)


def test_one_mark_credits_one_stud():
    names = sorted(SLOT_OF)
    M.TEST_SCREWS = screws(depth=0.0, caught=False)
    objects = all_on_marks(**{names[1]: on_marks(SLOT_OF[names[0]])})
    assert M.score(objects) == pytest.approx(2 * M.PLACED_CREDIT)


@pytest.mark.parametrize(
    "label,objects",
    [
        ("NaN pose", lambda: all_on_marks(stud_1=ObjectState(np.full(3, np.nan), np.array([0, 0, 0, 1.0]),
                                                             np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3)))),
        ("non-unit quaternion", lambda: all_on_marks(stud_1=on_marks(SLOT_OF["stud_1"], quat=[0, 0, 0, 1.2]))),
        ("missing stud", lambda: {k: v for k, v in all_on_marks().items() if k != "stud_3"}),
        ("outline not a 2x4", lambda: all_on_marks(stud_1=ObjectState(
            on_marks(SLOT_OF["stud_1"]).pos, on_marks(SLOT_OF["stud_1"]).quat_xyzw, np.zeros(3), np.zeros(3),
            on_marks(SLOT_OF["stud_1"]).low, on_marks(SLOT_OF["stud_1"]).high + 0.004))),
    ],
)
def test_fail_safes_void_the_trial(label, objects):
    M.TEST_SCREWS = screws()
    assert M.score(objects()) == 0.0, label
    assert not M.success(objects()), label


def test_distance_is_zero_when_fastened():
    assert M.distance(all_on_marks()) == pytest.approx(0.0, abs=1e-9)


# -- screw model (screw_runtime) -----------------------------------------------------------------


def _frames(driver_tip, stud_slots):
    """Every object the model reads: the right driver with its bit's tip at ``driver_tip`` along +x, the studs on
    the given marks (tight to the bottom plate), the rest far away."""
    still = (np.zeros(3), np.zeros(3))
    frames = {}
    for name in SR.DRIVERS:
        pos = np.asarray(driver_tip, float) if name == "driver_right" else np.array((0.0, 5.0, 0.0))
        frames[name] = (pos, np.eye(3), *still)
    for i, stud in enumerate(M.STUDS):
        slot = stud_slots.get(stud.name)
        c = np.array((M.SEATED_X, M.SLOT_Y[slot], M.SEATED_Z)) if slot is not None else np.array((0.0, 5.0 + i, 0.0))
        frames[stud.name] = (c, np.eye(3), *still)
    for screw in M.SCREWS:
        frames[screw.name] = (np.asarray(screw.pos, float), np.eye(3), *still)
    return frames


def _grips(driver_tip, squeeze):
    grip = np.asarray(driver_tip, float) + np.asarray(M.DRIVER_GRIP)
    target = M.HANDLE_DEPTH - (0.013 if squeeze else 0.008)
    return {"right": (grip, M.HANDLE_DEPTH, target), "left": (np.array((0.0, 5.0, 0.0)), 0.094, 0.094)}


def _drive(model, slot, stud_slots, squeeze=True, seconds=3.0, follow=True):
    """Hold the right driver's bit on screw ``slot``'s head, following it in when ``follow``."""
    dt = 1 / 240
    for _ in range(int(seconds / dt)):
        face = model.screw_pos(slot) - np.array((M.HEAD_T, 0.0, 0.0))
        tip = face if follow else np.asarray(M.SCREWS[slot].pos) - np.array((M.HEAD_T, 0.0, 0.0))
        model.update(_frames(tip, stud_slots), _grips(tip, squeeze), dt, {})


def test_the_trigger_needs_a_squeeze_not_just_a_hold():
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0}, squeeze=False)
    assert model.advance[0] == 0.0 and not model.trigger


def test_a_screw_driven_with_its_stud_on_the_marks_catches_and_fastens_it():
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0})
    assert model.caught[0] == "stud_3"
    assert model.seated(0) and model.fastened[0]


def test_a_screw_stops_when_the_bit_does_not_follow_the_head():
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0}, follow=False)
    assert 0.0 < model.advance[0] < 0.006  # the head draws ahead of the bit and the bit drops out
    assert not model.seated(0)


def test_a_screw_driven_with_no_stud_there_catches_nothing():
    model = SR.Model()
    _drive(model, 0, {})
    assert model.seated(0) and model.caught[0] is None and not model.fastened[0]


# -- demo --------------------------------------------------------------------------------------


def test_demo_uses_both_arms_and_fits_the_run_limit():
    plans = M.demo()
    assert set(plans) == {"left", "right"}
    longest = max(sum(w.duration for w in plans[arm]) for arm in plans)
    assert longest * 20 < 2400


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_is_reachable_with_the_official_ik(arm):
    plan = ArmPlan(list(M.demo()[arm]), station.BASES[arm])
    lo, hi = np.asarray(ik.LOWER), np.asarray(ik.UPPER)
    t = 0.0
    while t <= plan.duration:
        q = plan.action7(t)[0][:6]
        assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)
        t += 0.05
    assert plan.worst_ik_mm < 1.0
    assert plan.worst_ik_deg < 0.5
