"""Our model of the three framing screws and the two cordless impact drivers (pure NumPy; the simulator hook is
installed lazily on the GPU host only). Its simulator hook follows the screw model of the repository's lever-set task
(tasks/fit-lever-set-to-bored-door/lockset_runtime.py), and its contact reads follow the robot-cable contact monitor
of tasks/move-live-wire-with-stick (a PhysX rigid-contact view, read at every physics substep).

PhysX cannot cut a thread into wood, so the screws are modelled; everything the robot does to drive them is physical:
- Each screw starts through its pilot hole in the bottom plate, tip flush with the plate's inner face, head standing
  SCREW_PROUD proud. Screws are kinematic bodies: the model holds each on its axis and moves it only as it is driven
  (a driven screw goes only one way), and its head is a solid the driver's bit presses against.
- The trigger is a block on the front of the driver's grip. It is pulled while PhysX reports the robot pressing its
  face with at least F_TRIGGER along its travel; a lighter hold on the grip carries the driver without pulling it.
- The #2 square bit is in a screw's head once PhysX reports the bit's tip touching the head (a contact within TIP_ZONE
  of the tip) with the tip within RADIAL_IN of the screw's axis and the bit's axis within ANGLE_IN of the screw's. A
  light spring (the recess, at most F_RECESS_MAX) then holds the tip on the axis. Drawn AXIAL_OUT back from the head's
  face or pushed RADIAL_OUT off its axis, the bit is out.
- Driving: the screw turns in, at SPEED, only while the bit is in its head, the trigger is pulled, and the bit's tip
  is pressed into the head along the screw's axis with at least F_PUSH (PhysX contact force). Once the head draws
  away from the bit the push falls and the screw stops, so the hand has to follow the head in, pressing.
- Catching: as the screw's tip leaves the plate's inner face it catches the stud whose end is there (the screw's
  axis at least EDGE inside the end's edges, the stud square to the plate within CATCH_ANGLE); the thread then draws
  that stud's end toward the plate as the screw goes in (at most F_DRAW), and the shank holds the end where it caught.
  A screw driven with no stud end there goes on into the gap and holds nothing.
- Fastened: a caught screw driven until its head sits on the plate fixes the stud where it is.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys

import numpy as np

# Set by the task module (metres, world frame; the screws drive along +x).
SCREW_STARTS = ()  # each screw's head underside on its axis, as started
SCREW_PROUD = 0.0381  # how far a screw goes in, started to seated
PLATE_INNER_X = 0.226  # the bottom plate's inner face: where a screw's tip starts and a stud's end is drawn to
STUDS = ()  # stud names
SCREWS = ()  # screw body names, by mark
STUD_HALF = (0.1524, 0.01905, 0.04445)  # half length, thickness, depth
HEAD_T = 0.003  # the head's thickness: its face is this far out from its underside
BIT_R = 0.0032  # the bit's radius
TRIGGER_FACE = {}  # driver -> its trigger's face in its own frame: x, (y0, y1), (z0, z1)
DRIVERS = ("driver",)

AXIS = np.array((1.0, 0.0, 0.0))
SPEED = 0.025  # m/s: a 3 in screw goes in in about 1.5 s
F_PUSH = 20.0  # N: the bit pressed into the head along the screw's axis, at least this, for the screw to turn
F_TRIGGER = 17.0  # N: pressed with at least this along its travel, the trigger is pulled (a power tool's trigger
# takes ~10-20 N); with the 20 N push, the pad behind the grip carries ~37 N, within the YAM's ~40 N per finger
TIP_ZONE = 0.003  # a contact within this of the bit's tip (behind it, and around the bit) is the tip's
TRIGGER_ZONE = 0.003  # a contact within this of the trigger's face (and its edges) is on the trigger
RADIAL_IN, RADIAL_OUT = 0.003, 0.005  # the bit's tip from the screw's axis: to go in, and to stay in
AXIAL_OUT = 0.002  # the tip drawn back this far from the head's face: out of the recess
ANGLE_IN, ANGLE_OUT = 8.0, 15.0  # degrees between the bit's axis and the screw's
K_RECESS, C_RECESS, F_RECESS_MAX = 1000.0, 10.0, 5.0  # N/m, N s/m, N: the recess centring the bit
EDGE = 0.006  # the screw's axis at least this far inside the stud end's edges to catch it
CATCH_ANGLE = math.radians(10.0)
CATCH_WINDOW = 0.003  # the stud's end within this of the screw's tip as the tip crosses it
K_DRAW, F_DRAW = 20000.0, 150.0  # N/m, N: the thread drawing the stud's end toward the plate
K_SHANK, C_SHANK, F_SHANK = 4000.0, 10.0, 30.0  # the shank in the stud's end grain, across the screw
NAMES = ()  # every object the model reads, set by the task module
ROBOT = "robot"  # what the contact reads call any robot link


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
        self.trigger = {d: 0.0 for d in DRIVERS}  # each trigger's press (N), from the last contacts
        self.push = [0.0] * n  # each screw's push from a bit's tip (N), from the last contacts
        self.peak_push = [0.0] * n  # the most push each screw saw while it turned (for the logs)
        self.min_push = [math.inf] * n  # the least push each screw saw while it turned

    def seated(self, i) -> bool:
        return self.advance[i] >= SCREW_PROUD - 1e-6

    def screw_pos(self, i):
        return np.asarray(SCREW_STARTS[i], float) + AXIS * self.advance[i]

    @staticmethod
    def classify(Rd, pd, hits, face):
        """From one driver's contacts [(other, point, normal, force N), ...] in the environment frame: the trigger's
        press (N along its travel) and, per screw, the tip's push along the screw's axis (N) and whether the tip
        touches it."""
        trigger = 0.0
        push = [0.0] * len(SCREWS)
        touch = [False] * len(SCREWS)
        x_face, (y0, y1), (z0, z1) = face
        for other, p, nrm, f in hits:
            # PhysX reports each contact's normal force signed along its normal, which for some pairs points into
            # the sensing body (the pads on a driver's grip read negative, its bit on a head positive): use its size.
            f = abs(f)
            if f <= 1e-6:
                continue  # inside the contact offset, not pressing
            local = Rd.T @ (np.asarray(p, float) - pd)
            nrm = np.asarray(nrm, float)
            if other in SCREWS:
                if -TIP_ZONE <= local[0] <= TIP_ZONE and math.hypot(local[1], local[2]) <= BIT_R + TIP_ZONE:
                    i = SCREWS.index(other)
                    touch[i] = True
                    push[i] += f * abs(float(nrm @ AXIS))
            elif other == ROBOT:
                if (abs(local[0] - x_face) <= TRIGGER_ZONE and y0 - TRIGGER_ZONE <= local[1] <= y1 + TRIGGER_ZONE
                        and z0 - TRIGGER_ZONE <= local[2] <= z1 + TRIGGER_ZONE):
                    trigger += f * abs(float(nrm @ Rd[:, 0]))
        return trigger, push, touch

    def update(self, frames, contacts, dt, acc):
        """frames: name -> (pos, rot, vel, spin) in the environment frame; contacts: driver -> [(other, point,
        normal, force), ...] from the last physics substep, ``other`` a screw's name or ROBOT."""
        if any(n not in frames for n in NAMES):
            return
        self.push = [0.0] * len(SCREW_STARTS)
        for drv in DRIVERS:
            pd, Rd, vd, wd = frames[drv]
            trig_f, push, touch = self.classify(Rd, pd, contacts.get(drv, ()), TRIGGER_FACE[drv])
            self.trigger[drv] = trig_f
            pulled = trig_f >= F_TRIGGER
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
                    if touch[i] and radial <= RADIAL_IN and ang <= ANGLE_IN:
                        self.engaged[i] = drv
                elif radial > RADIAL_OUT or axial < -AXIAL_OUT or ang > ANGLE_OUT:
                    self.engaged[i] = None
                if self.engaged[i] == drv:
                    self.push[i] = push[i]
                    v_rad = vd - (vd @ AXIS) * AXIS
                    f = -K_RECESS * radial_v - C_RECESS * v_rad
                    n = float(np.linalg.norm(f))
                    if n > F_RECESS_MAX:
                        f *= F_RECESS_MAX / n
                    _wrench(acc, drv, pd, f, tip)
                    if pulled and touch[i] and push[i] >= F_PUSH and not self.seated(i):
                        self.advance[i] = min(SCREW_PROUD, self.advance[i] + SPEED * dt)
                        self.peak_push[i] = max(self.peak_push[i], push[i])
                        self.min_push[i] = min(self.min_push[i], push[i])
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
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


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


# Contact reports: Isaac Lab switches PhysX contact processing off unless one of its ContactSensors exists (this
# embodiment builds none), so it is switched back on, as Isaac Lab's own ContactSensor does, when the physics is
# configured (as tasks/move-live-wire-with-stick does).
_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"
_VIEW = {"view": None, "sensors": (), "filters": (), "err": None}
_DEBUG = bool(__import__("os").environ.get("SCREW_DEBUG"))


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr._configure_physics, "_screw_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        try:
            import carb

            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        except Exception as e:  # noqa: BLE001
            print(f"SCREW_CONTACTS could not switch contact processing on: {e!r}", flush=True)

    _configure_physics._screw_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)


class _MgrFinder(importlib.abc.MetaPathFinder):
    """Patches Isaac Lab's PhysxManager when it is first imported (it needs Kit, so not at task import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _PHYSX_MGR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_module
        return spec


def prepare() -> None:
    """Called at task import: arrange for contact processing to be on when the physics starts."""
    if _PHYSX_MGR in sys.modules:
        _patch_manager(sys.modules[_PHYSX_MGR])
    elif not any(isinstance(f, _MgrFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MgrFinder())


def _make_view():
    """A rigid-contact view: sensors the two drivers (contact reporting is applied to them at spawn), filters the
    screws and every robot link."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    mgr = sys.modules[_PHYSX_MGR].PhysxManager
    sv = mgr.get_physics_sim_view()
    if sv is None:
        return
    stage = get_current_stage()
    drivers, screws, robot = {}, {}, []
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        leaf = p.rsplit("/", 1)[-1]
        if leaf in DRIVERS:
            drivers[leaf] = p
        elif leaf in SCREWS:
            screws[leaf] = p
        elif "/LeftArm/" in p or "/RightArm/" in p:
            robot.append(p)
    if len(drivers) != len(DRIVERS) or len(screws) != len(SCREWS) or not robot:
        raise RuntimeError(f"screw contacts: {len(drivers)} drivers, {len(screws)} screws, {len(robot)} robot links")
    sensors = [drivers[d] for d in DRIVERS]
    filters = [screws[s] for s in SCREWS] + robot
    view = sv.create_rigid_contact_view(sensors, filter_patterns=[filters] * len(sensors),
                                        max_contact_data_count=64 * len(sensors) * len(filters))
    _VIEW.update(view=view, sensors=tuple(DRIVERS), filters=tuple(SCREWS) + (ROBOT,) * len(robot))
    print(f"SCREW_CONTACTS view: {len(sensors)} drivers x ({len(screws)} screws + {len(robot)} robot links)", flush=True)


def _contacts(origin, dt):
    """driver -> [(screw name or ROBOT, point, normal, force N), ...] from the last physics substep."""
    try:
        if _VIEW["view"] is None:
            _make_view()
        view = _VIEW["view"]
        if view is None:
            return {}
        forces, points, normals, _, counts, starts = view.get_contact_data(dt=dt)
    except Exception as e:  # noqa: BLE001
        if _VIEW["err"] is None:
            _VIEW["err"] = repr(e)
            print(f"SCREW_CONTACTS error: {e!r}", flush=True)
        return {}
    counts = _np(counts).astype(int).reshape(len(_VIEW["sensors"]), len(_VIEW["filters"]))
    if _DEBUG:
        _VIEW["polls"] = _VIEW.get("polls", 0) + 1
        if _VIEW["polls"] % 120 == 0:
            st, fo, se = _np(starts).astype(int).reshape(counts.shape), _np(forces).reshape(-1), _np(_).reshape(-1)
            rows = []
            for s in range(counts.shape[0]):
                for k in range(counts.shape[1]):
                    if counts[s, k]:
                        sl = slice(st[s, k], st[s, k] + counts[s, k])
                        rows.append((s, k, int(counts[s, k]), np.round(fo[sl], 2).tolist()[:4],
                                     np.round(se[sl] * 1000, 2).tolist()[:4]))
            print(f"SCREW_RAW poll {_VIEW['polls']} shape {counts.shape} pairs(sensor, filter, n, forces, sep mm): "
                  f"{rows}", flush=True)
    if not counts.any():
        return {}
    starts = _np(starts).astype(int).reshape(counts.shape)
    forces, points, normals = _np(forces).reshape(-1), _np(points).reshape(-1, 3), _np(normals).reshape(-1, 3)
    out = {}
    for s, drv in enumerate(_VIEW["sensors"]):
        hits = []
        for k, other in enumerate(_VIEW["filters"]):
            for j in range(starts[s, k], starts[s, k] + counts[s, k]):
                hits.append((other, points[j] - origin, normals[j], float(forces[j])))
        out[drv] = hits
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
    _VIEW["view"] = None  # rebuilt on the next substep (the environment may have been rebuilt)
    if getattr(anchor, "_screw_hooked", False):
        return
    orig = anchor.write_data_to_sim
    origin = _np(base.scene.env_origins)[0][:3]
    start_raw = {}
    held = {}
    log = {"n": 0}

    def write():
        orig()
        frames, raw = {}, {}
        for n, o in objs.items():
            pose = _np(o.data.root_link_pose_w)[0]
            vel = _np(o.data.root_link_vel_w)[0]
            raw[n] = pose[:7].copy()
            frames[n] = (pose[:3] - origin, _quat_matrix(pose[3:7]), vel[:3], vel[3:6])
        for n in SCREWS:
            start_raw.setdefault(n, raw[n].copy())  # the started screws' poses (orientation never changes)
        if not any(MODEL.advance) and not any(MODEL.fastened):
            held.clear()
        acc = {}
        before = list(MODEL.advance)
        contacts = _contacts(origin, PHYSICS_DT)
        MODEL.update(frames, contacts, PHYSICS_DT, acc)
        log["n"] += 1
        if _DEBUG and log["n"] % 120 == 0:
            for drv, hits in contacts.items():
                pd, Rd = frames[drv][0], frames[drv][1]
                pts = [(np.round((Rd.T @ (np.asarray(p) - pd)) * 1000, 1).tolist(), round(f, 1),
                        np.round(Rd.T @ np.asarray(nrm), 2).tolist()) for other, p, nrm, f in hits
                       if other == ROBOT and f > 0.0]
                if pts:
                    print(f"SCREW_DEBUG {drv} robot contacts (driver frame mm, N, normal): {pts[:8]}", flush=True)
        if log["n"] % 24 == 0 and (any(MODEL.engaged) or any(f >= 5.0 for f in MODEL.trigger.values())):
            moving = [i + 1 for i in range(len(SCREWS)) if MODEL.advance[i] > before[i]]
            print("SCREW_STATE " + " ".join(f"trigger_{d.split('_')[-1]} {f:.1f}N" for d, f in MODEL.trigger.items())
                  + " " + " ".join(f"screw{i + 1} {MODEL.engaged[i] or '-'} push {MODEL.push[i]:.1f}N "
                                   f"in {MODEL.advance[i] * 1000:.1f}mm" for i in range(len(SCREWS)))
                  + f" turning {moving}", flush=True)
        dev = base.device
        pins = {}
        for i, n in enumerate(SCREWS):
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
            if n not in SCREWS:  # the screws are kinematic: no velocity to set
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
