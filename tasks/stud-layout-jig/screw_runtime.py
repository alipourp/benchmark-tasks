"""Our model of the three framing screws and the two cordless impact drivers (pure NumPy; the simulator hook is
installed lazily on the GPU host only). Its bit-in-head check and its simulator hook follow the screw model of the
repository's lever-set task (tasks/fit-lever-set-to-bored-door/lockset_runtime.py).

PhysX cannot cut a thread into wood, so the screws are modelled:
- Each screw starts through its pilot hole in the bottom plate, tip flush with the plate's inner face, head standing
  SCREW_PROUD proud. The model holds each screw on its axis and moves it only as it is driven (a driven screw goes
  only one way; nothing else moves it).
- An impact driver's trigger is pressed while a hand squeezes its grip, the pads on the grip's front (the trigger)
  and back: the hand's TCP within TRIGGER_REACH of the grip point, the pads on the grip, and the gripper driven at
  least TRIGGER_SQUEEZE past them (a squeeze); a lighter hold carries the driver without pulling the trigger.
- The #2 square bit is in a screw's head when its tip is within RADIAL_IN of the screw's axis, at the head's face
  (AXIAL_IN short of it to AXIAL_OUT past it) and its axis within ANGLE_IN of the screw's; a light spring (the
  recess, at most F_RECESS_MAX) then holds the tip on the axis. Drawn AXIAL_OUT back from the head or pushed
  RADIAL_OUT off its axis, the bit comes out.
- Driving: while the bit is in the head and the trigger is pressed, the screw goes in at SPEED: once its thread is
  in the wood it draws itself in, and the hand has to follow the head in to keep the bit in it. If the head draws
  more than AXIAL_OUT ahead of the bit, the bit is out and the screw stops until the bit is back in.
- Catching: as the screw's tip leaves the plate's inner face it catches the stud whose end is there (the screw's
  axis at least EDGE inside the end's edges, the stud square to the plate within CATCH_ANGLE); the thread then draws
  that stud's end toward the plate as the screw goes in (at most F_DRAW), and the shank holds the end where it caught.
  A screw driven with no stud end there goes on into the gap and holds nothing.
- Fastened: a caught screw driven until its head sits on the plate fixes the stud where it is.
"""
from __future__ import annotations

import math

import numpy as np

# Set by the task module (metres, world frame; the screws drive along +x).
SCREW_STARTS = ()  # each screw's head underside on its axis, as started
SCREW_PROUD = 0.0381  # how far a screw goes in, started to seated
PLATE_INNER_X = 0.226  # the bottom plate's inner face: where a screw's tip starts and a stud's end is drawn to
STUDS = ()  # stud names
STUD_HALF = (0.1524, 0.01905, 0.04445)  # half length, thickness, depth
DRIVER_GRIP = (0.0, 0.0, 0.0)  # the handle's grip point in the driver's frame
HANDLE_DEPTH = 0.045  # front (trigger) to back of the handle, where the pads close
HEAD_T = 0.003  # the head's thickness: its face is this far out from its underside
DRIVERS = ("driver",)

AXIS = np.array((1.0, 0.0, 0.0))
SPEED = 0.025  # m/s: a 3 in screw goes in in about 1.5 s
TRIGGER_REACH, TRIGGER_SLACK, TRIGGER_SQUEEZE = 0.040, 0.0015, 0.012  # the pads anywhere on the grip's middle
RADIAL_IN, RADIAL_OUT = 0.003, 0.005  # the bit's tip from the screw's axis: to go in, and to stay in
AXIAL_IN, AXIAL_OUT = 0.003, 0.004  # the tip short of the head's face: to go in, and to stay in (a bit resting on
# a head reads ~1.5 mm short: the colliders' contact offsets)
ANGLE_IN, ANGLE_OUT = 8.0, 15.0  # degrees between the bit's axis and the screw's
K_RECESS, C_RECESS, F_RECESS_MAX = 1000.0, 10.0, 5.0  # N/m, N s/m, N: the recess centring the bit
EDGE = 0.006  # the screw's axis at least this far inside the stud end's edges to catch it
CATCH_ANGLE = math.radians(10.0)
CATCH_WINDOW = 0.003  # the stud's end within this of the screw's tip as the tip crosses it
K_DRAW, F_DRAW = 20000.0, 150.0  # N/m, N: the thread drawing the stud's end toward the plate
K_SHANK, C_SHANK, F_SHANK = 4000.0, 10.0, 30.0  # the shank in the stud's end grain, across the screw
NAMES = ()  # every object the model reads, set by the task module


class Model:
    def __init__(self):
        self.clear()

    def clear(self):
        n = len(SCREW_STARTS)
        self.advance = [0.0] * n
        self.engaged = [None] * n  # the driver whose bit is in each screw
        self.caught = [None] * n
        self.gap_catch = [0.0] * n
        self.adv_catch = [0.0] * n
        self.lateral = [None] * n
        self.fastened = [False] * n
        self.trigger = False

    def seated(self, i) -> bool:
        return self.advance[i] >= SCREW_PROUD - 1e-6

    def screw_pos(self, i):
        return np.asarray(SCREW_STARTS[i], float) + AXIS * self.advance[i]

    def update(self, frames, grips, dt, acc):
        """frames: name -> (pos, rot, vel, spin) in the environment frame; grips: arm -> (tcp_pos, pad gap, the gap
        the gripper is driven to)."""
        if any(n not in frames for n in NAMES):
            return
        self.trigger = False
        for drv in DRIVERS:
            pd, Rd, vd, wd = frames[drv]
            grip = pd + Rd @ np.asarray(DRIVER_GRIP)
            trigger = any(
                np.linalg.norm(np.asarray(tcp) - grip) <= TRIGGER_REACH and gap <= HANDLE_DEPTH + TRIGGER_SLACK
                and target <= HANDLE_DEPTH - TRIGGER_SQUEEZE
                for tcp, gap, target in grips.values()
            )
            self.trigger = self.trigger or trigger
            tip, bit = pd, Rd[:, 0]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(bit @ AXIS)))))
            for i in range(len(SCREW_STARTS)):
                if self.engaged[i] not in (None, drv):
                    continue  # another driver's bit is in this screw
                face = self.screw_pos(i) - AXIS * HEAD_T
                d = tip - face
                axial = float(d @ AXIS)  # > 0: the tip is past the head's face
                radial_v = d - axial * AXIS
                radial = float(np.linalg.norm(radial_v))
                if self.engaged[i] is None:
                    if radial <= RADIAL_IN and -AXIAL_IN <= axial <= AXIAL_OUT and ang <= ANGLE_IN:
                        self.engaged[i] = drv
                elif radial > RADIAL_OUT or axial < -AXIAL_OUT or ang > ANGLE_OUT:
                    self.engaged[i] = None
                if self.engaged[i] == drv:
                    v_rad = vd - (vd @ AXIS) * AXIS
                    f = -K_RECESS * radial_v - C_RECESS * v_rad
                    n = float(np.linalg.norm(f))
                    if n > F_RECESS_MAX:
                        f *= F_RECESS_MAX / n
                    _wrench(acc, drv, pd, f, tip)
                    if trigger and not self.seated(i):
                        self.advance[i] = min(SCREW_PROUD, self.advance[i] + SPEED * dt)
        for i in range(len(SCREW_STARTS)):
            self._thread(i, frames, acc)

    def _thread(self, i, frames, acc):
        tip_x = PLATE_INNER_X + self.advance[i]
        axis_pt = np.asarray(SCREW_STARTS[i], float)
        if self.caught[i] is None and self.advance[i] > 0.0:
            for name in STUDS:
                c, R, _, _ = frames[name]
                end = _near_end(c, R)
                if abs(float(R[:, 0] @ AXIS)) < math.cos(CATCH_ANGLE):
                    continue
                if not (-CATCH_WINDOW <= tip_x - float(end @ AXIS) <= CATCH_WINDOW):
                    continue
                rel = axis_pt - end
                rel = rel - (rel @ AXIS) * AXIS
                if abs(float(rel @ R[:, 1])) <= STUD_HALF[1] - EDGE and abs(float(rel @ R[:, 2])) <= STUD_HALF[2] - EDGE:
                    self.caught[i] = name
                    self.gap_catch[i] = max(0.0, float(end @ AXIS) - PLATE_INNER_X)
                    self.adv_catch[i] = self.advance[i]
                    self.lateral[i] = rel
                    break
        name = self.caught[i]
        if name is None or self.fastened[i]:
            if name is not None and self.seated(i):
                self.fastened[i] = True
            return
        c, R, v, w = frames[name]
        end = _near_end(c, R)
        # the thread draws the stud's end toward the plate as the screw goes in
        allowed = max(0.0, self.gap_catch[i] - (self.advance[i] - self.adv_catch[i]))
        s = float(end @ AXIS) - PLATE_INNER_X
        f_draw = min(F_DRAW, K_DRAW * max(0.0, s - allowed))
        if f_draw > 0.0:
            _wrench(acc, name, c, -f_draw * AXIS, end)
        # the shank holds the end where the screw caught it, across the screw's axis
        rel = axis_pt - end
        rel = rel - (rel @ AXIS) * AXIS
        drift = rel - self.lateral[i]
        v_end = v + np.cross(w, end - c)
        f = K_SHANK * drift - C_SHANK * (v_end - (v_end @ AXIS) * AXIS)
        n = float(np.linalg.norm(f))
        if n > F_SHANK:
            f *= F_SHANK / n
        _wrench(acc, name, c, f, end)
        if self.seated(i):
            self.fastened[i] = True


def _near_end(c, R):
    """The centre of the stud's end nearer the bottom plate."""
    a, b = c - R[:, 0] * STUD_HALF[0], c + R[:, 0] * STUD_HALF[0]
    return a if a @ AXIS < b @ AXIS else b


def _wrench(acc, name, origin, force, at):
    """Force (N, world) at a world point, as a force plus a torque about the body's origin (as the lever-set task's model does)."""
    f, t, o = acc.setdefault(name, [np.zeros(3), np.zeros(3), np.asarray(origin, float)])
    f += force
    t += np.cross(np.asarray(at, float) - o, force)


MODEL = Model()


# ---- simulator hook (GPU host only), after the lever-set task's --------------------------------------------------------------
def _np(x):
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy(), dtype=np.float64)


def _quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _grips(base, origin):
    """Each arm's TCP (environment frame), the gap between its pads and the gap the gripper is driven to."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = {}
    for arm in ("left", "right"):
        art = base.scene[arm + "_robot"]
        q = _np(art.data.joint_pos)[0]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        tcp = ik.tcp_pose(np.asarray(q)[arm_ids], station.BASES[arm])[0]
        q_target = _np(art.data.joint_pos_target)[0]
        gap = float(sum(abs(q[k]) for k in finger_ids))  # each finger runs from 0 (closed) to 47 mm open
        target = float(sum(abs(q_target[k]) for k in finger_ids))
        out[arm] = (np.asarray(tcp, float), gap, target)
    return out


def install(environment):
    """Run the model before every physics substep (Isaac Lab calls ``write_data_to_sim`` then) and apply its forces;
    hold each screw where the model has it and each fastened stud where it was fastened. Idempotent."""
    import torch
    import warp as wp
    from yam_arena.constants import PHYSICS_DT

    base = environment.unwrapped
    objs = {n: base.scene.rigid_objects[n] for n in NAMES}
    anchor = objs[DRIVERS[0]]
    if getattr(anchor, "_screw_hooked", False):
        return
    orig = anchor.write_data_to_sim
    origin = _np(base.scene.env_origins)[0][:3]
    start_raw = {}
    held = {}
    screws = [f"screw_{i + 1}" for i in range(len(SCREW_STARTS))]

    def write():
        orig()
        frames, raw = {}, {}
        for n, o in objs.items():
            pose = _np(o.data.root_link_pose_w)[0]
            vel = _np(o.data.root_link_vel_w)[0]
            raw[n] = pose[:7].copy()
            frames[n] = (pose[:3] - origin, _quat_matrix(pose[3:7]), vel[:3], vel[3:6])
        for n in screws:
            start_raw.setdefault(n, raw[n].copy())  # the started screws' poses (orientation never changes)
        if not any(MODEL.advance) and not any(MODEL.fastened):
            held.clear()
        acc = {}
        MODEL.update(frames, _grips(base, origin), PHYSICS_DT, acc)
        dev = base.device
        pins = {}
        for i, n in enumerate(screws):
            pose = start_raw[n].copy()
            pose[:3] = MODEL.screw_pos(i) + origin
            pins[n] = pose
        for i, name in enumerate(MODEL.caught):
            if MODEL.fastened[i] and name is not None:
                held.setdefault(name, raw[name].copy())
        pins.update(held)
        for n, pose in pins.items():
            objs[n].write_root_link_pose_to_sim_index(
                root_pose=torch.tensor([pose.tolist()], dtype=torch.float32, device=dev))
            objs[n].write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
            acc.pop(n, None)
        for n, (f, t, o_body) in acc.items():
            if not f.any() and not t.any():
                continue
            o = objs[n]
            com = _np(o.data.body_com_pos_w)[0].reshape(-1)[:3] - origin if hasattr(o.data, "body_com_pos_w") else o_body
            t_com = t + np.cross(o_body - com, f)  # move the torque from the origin to the centre of mass
            o.root_view.apply_forces_and_torques_at_position(
                force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                torque_data=wp.array(t_com.astype(np.float32), dtype=wp.float32, device=dev),
                position_data=None, indices=o._ALL_INDICES, is_global=True)

    anchor.write_data_to_sim = write
    anchor._screw_hooked = True
