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
TOUCH = 0.0005  # offline stand-in for PhysX: the bit's tip presses a head only within this of its face


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


def _trigger_point(driver_tip):
    return np.asarray(driver_tip, float) + np.array((M.TRIGGER_FACE_X, M.DRIVER_SIDE["right"] * M.TRIGGER_Y, 0.0))


def _contacts(model, driver_tip, slot, push, trigger):
    """What PhysX would report for the right driver: the bit's tip on screw ``slot``'s head with ``push`` N (only
    while the tip is at the head's face), and a pad on the trigger's face with ``trigger`` N."""
    tip = np.asarray(driver_tip, float)
    face = model.screw_pos(slot) - np.array((M.HEAD_T, 0.0, 0.0))
    hits = []
    if abs(float(tip[0] - face[0])) <= TOUCH and push > 0.0:
        hits.append((SR.SCREWS[slot], tip, np.array((-1.0, 0.0, 0.0)), push))
    if trigger > 0.0:
        hits.append((SR.ROBOT, _trigger_point(tip), np.array((-1.0, 0.0, 0.0)), trigger))
    return {"driver_right": hits}


def _drive(model, slot, stud_slots, push=25.0, trigger=35.0, seconds=3.0, follow=True, short=0.0):
    """Hold the right driver's bit on screw ``slot``'s head (``short`` m short of it), pressing with ``push`` N and
    the trigger with ``trigger`` N, following the head in when ``follow``."""
    dt = 1 / 240
    for _ in range(int(seconds / dt)):
        face = model.screw_pos(slot) - np.array((M.HEAD_T, 0.0, 0.0))
        tip = face if follow else np.asarray(M.SCREWS[slot].pos) - np.array((M.HEAD_T, 0.0, 0.0))
        tip = tip - np.array((short, 0.0, 0.0))
        model.update(_frames(tip, stud_slots), _contacts(model, tip, slot, push, trigger), dt, {})


def test_a_screw_driven_with_its_stud_on_the_marks_catches_and_fastens_it():
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0})
    assert model.caught[0] == "stud_3"
    assert model.seated(0) and model.fastened[0]
    assert model.min_push[0] >= SR.F_PUSH


def test_contact_forces_count_by_size_whatever_their_sign():
    """PhysX reports the pads' forces on the grip as negative (along a normal into the driver): they still pull the
    trigger and drive the screw (the earlier reader skipped them, so the trigger never read more than 0 N)."""
    model = SR.Model()
    dt = 1 / 240
    for _ in range(int(3.0 / dt)):
        tip = model.screw_pos(0) - np.array((M.HEAD_T, 0.0, 0.0))
        hits = {d: [(o, p, n, -f) for o, p, n, f in h] for d, h in _contacts(model, tip, 0, 25.0, 35.0).items()}
        model.update(_frames(tip, {"stud_3": 0}), hits, dt, {})
    assert model.trigger["driver_right"] == pytest.approx(35.0)
    assert model.seated(0) and model.fastened[0]


def test_the_trigger_needs_a_squeeze_not_just_a_hold():
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0}, trigger=15.0)  # the carrying hold, ~15 N on the trigger
    assert model.advance[0] == 0.0
    assert model.trigger["driver_right"] == pytest.approx(15.0)


@pytest.mark.parametrize("push,turns", [(0.0, False), (10.0, False), (19.5, False), (20.0, True), (30.0, True)])
def test_a_screw_turns_only_with_the_bit_pressed_in_with_20_newtons(push, turns):
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0}, push=push, seconds=1.0)
    assert (model.advance[0] > 0.0) == turns


@pytest.mark.parametrize("short", [0.001, 0.003])
def test_a_bit_short_of_the_head_does_not_drive_it(short):
    """No contact, no drive: the bit must touch the head (the earlier model let it drive from up to 3 mm short)."""
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0}, short=short, seconds=1.0)
    assert model.advance[0] == 0.0 and model.engaged[0] is None


def test_contacts_off_the_bit_tip_or_the_trigger_do_not_count():
    """The chuck against a head, or a pad on the grip's back, are not the bit in the head or the trigger pulled."""
    model = SR.Model()
    tip = np.asarray(M.SCREWS[0].pos) - np.array((M.HEAD_T, 0.0, 0.0))
    chuck = tip + np.array((-(M.BIT_LEN + 0.005), 0.0, 0.0))
    back = tip + np.array((M.HANDLE_X - M.HANDLE_DEPTH / 2, M.DRIVER_SIDE["right"] * M.TRIGGER_Y, 0.0))
    hits = {"driver_right": [(SR.SCREWS[0], chuck, np.array((-1.0, 0.0, 0.0)), 40.0),
                             (SR.ROBOT, back, np.array((1.0, 0.0, 0.0)), 40.0)]}
    for _ in range(240):
        model.update(_frames(tip, {"stud_3": 0}), hits, 1 / 240, {})
    assert model.advance[0] == 0.0 and model.engaged[0] is None
    assert model.trigger["driver_right"] == 0.0


def test_a_screw_stops_when_the_bit_does_not_follow_the_head():
    model = SR.Model()
    _drive(model, 0, {"stud_3": 0}, follow=False)
    assert 0.0 < model.advance[0] < 0.003  # the head draws away from the bit: no contact, no push, no drive
    assert not model.seated(0)


def test_a_screw_driven_with_no_stud_there_catches_nothing():
    model = SR.Model()
    _drive(model, 0, {})
    assert model.seated(0) and model.caught[0] is None and not model.fastened[0]


def test_the_trigger_sits_where_the_pads_close_and_the_hold_and_squeeze_straddle_it():
    """The hand's grip point is midway between the trigger's face and the grip's back, at the trigger; the hold
    presses it below the trigger's 17 N and the squeeze above it, even while pushing the demo's 23 N."""
    assert M.TRIGGER_FACE_X - M.DRIVER_GRIP[0] == pytest.approx(M.GRIP_SPAN / 2)
    assert abs(M.DRIVER_GRIP[1] - M.TRIGGER_Y) < M.TRIGGER_LEN / 2 - 0.005
    hold = M.FINGER_K * (M.GRIP_SPAN / 2 + M.FINGER_HOLD)
    squeeze = M.FINGER_K * (M.GRIP_SPAN / 2 + M.FINGER_SQUEEZE)
    # pushing loads the back pad and unloads the trigger by half the push; the simulated finger tops out at 50 N
    assert hold < SR.F_TRIGGER < min(squeeze - M.PUSH / 2, 50.0 - M.PUSH)
    assert SR.F_TRIGGER + SR.F_PUSH <= 40.0  # a real trial: the back pad within the YAM's ~40 N per finger
    assert 2 * 0.5 * hold > 1.5 * M.DRIVER_MASS * 9.81  # the hold carries the driver (pad friction ~0.5)
    assert SR.F_PUSH < M.PUSH


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
