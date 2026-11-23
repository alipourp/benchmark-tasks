"""jointkit: jointed bodies for yam_arena task files.

yam_arena task-file objects are single loose or fixed primitives. This module adds:

* :class:`Body`: one rigid body built from several boxes and upright cylinders (some visual only), loose or kinematic;
* :func:`hinge` and :func:`slider`: a PhysX revolute or prismatic joint between two Bodies, each with a spring/damper
  drive the task file can retune while an episode runs (:func:`set_drive`);
* :func:`on_reset` and :func:`on_step`: callbacks after every episode reset and every control step, so a task file can
  follow what happens over an episode, not only how it ends.

Keep this file next to the task file and import it from there. Importing it hooks ``yam_arena.scene`` and the
embodiment when they load, without changing yam_arena: a Body is spawned as one rigid body, its state handed to
``success()``/``score()`` is its origin's pose, and jointed Bodies are left out of the per-object start jitter (which
would pull the joints apart). Everything that needs Isaac Sim is imported lazily, so the task file loads without it.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, ObjectState

Color = tuple[float, float, float]


@dataclass(frozen=True)
class BoxPart:
    """A box in the body frame, ``centre`` and ``size`` (m), turned ``yaw`` (rad) about the body's z axis."""

    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    color: Color = (0.6, 0.6, 0.6)
    yaw: float = 0.0
    collide: bool = True

    def half_extent(self) -> np.ndarray:
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        sx, sy, sz = (v / 2 for v in self.size)
        return np.array([c * sx + s * sy, s * sx + c * sy, sz])


@dataclass(frozen=True)
class CylPart:
    """An upright cylinder in the body frame."""

    centre: tuple[float, float, float]
    radius: float
    height: float
    color: Color = (0.6, 0.6, 0.6)
    collide: bool = True

    def half_extent(self) -> np.ndarray:
        return np.array([self.radius, self.radius, self.height / 2])


@dataclass(frozen=True)
class Body(Box):
    """A rigid body made of ``parts``. A Box subclass only so yam_arena's task-file checks accept it: ``pos`` is the
    body origin (world, m), ``size`` its bounding box. ``kinematic`` bodies stay where they are put. ``ignore`` names
    bodies (earlier in OBJECTS) this one never collides with."""

    parts: tuple = ()
    kinematic: bool = False
    friction: tuple[float, float] = (0.8, 0.7)
    ignore: tuple[str, ...] = ()
    iterations: tuple[int, int] = (16, 1)  # PhysX solver position and velocity iterations: raise for jointed chains

    @staticmethod
    def build(name: str, pos, mass: float, parts, **kw) -> "Body":
        lo = np.min([np.array(p.centre) - p.half_extent() for p in parts], axis=0)
        hi = np.max([np.array(p.centre) + p.half_extent() for p in parts], axis=0)
        return Body(name=name, pos=tuple(float(v) for v in pos), size=tuple((hi - lo).tolist()), mass=mass,
                    parts=tuple(parts), **kw)

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min([np.array(p.centre) - p.half_extent() for p in self.parts], axis=0)
        hi = np.max([np.array(p.centre) + p.half_extent() for p in self.parts], axis=0)
        return lo, hi

    def _check(self) -> None:
        super()._check()
        if not self.parts:
            raise ValueError(f"{self.name}: a Body needs at least one part")
        if self.mass is None:
            raise ValueError(f"{self.name}: give a Body a mass (use kinematic=True to fix it in place)")


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(quat_xyzw) -> float:
    """Turn about the vertical (rad) of an orientation."""
    r = rotation(quat_xyzw)
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def body_state(spec: Body, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> ObjectState:
    """An ObjectState for a Body at a pose: ``pos`` its origin, bounds from its parts."""
    lo, hi = spec.local_bounds()
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    pts = np.asarray(pos, float) + corners @ rotation(quat_xyzw).T
    return ObjectState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat_xyzw, float),
                       lin_vel=np.asarray(lin_vel, float), ang_vel=np.asarray(ang_vel, float),
                       low=pts.min(0), high=pts.max(0))


# -- joints -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Joint:
    """A revolute (``kind="hinge"``, drive in degrees) or prismatic (``kind="slider"``, drive in m) joint letting
    ``body1`` turn about / slide along ``axis`` ("X", "Y" or "Z") through ``anchor`` (world, m, at the start; both
    bodies start unrotated) against ``body0``, between ``limits`` (from the start), with a spring/damper drive toward
    ``target``. ``body0`` must come before ``body1`` in OBJECTS. Every reset puts the drive back as declared."""

    name: str
    kind: str
    body0: str
    body1: str
    anchor: tuple[float, float, float]
    axis: str
    limits: tuple[float, float] | None
    stiffness: float
    damping: float
    target: float
    max_force: float


_SPECS: dict[str, Body] = {}
_JOINTS: list[Joint] = []
_PATHS: dict[str, str] = {}
_DRIVES: dict[str, tuple] = {}


def _add_joint(j: Joint) -> Joint:
    """Register ``j``, replacing one of the same name (a task file can be imported more than once)."""
    _JOINTS[:] = [k for k in _JOINTS if k.name != j.name]
    _JOINTS.append(j)
    return j


def hinge(name, body0, body1, anchor, axis="Z", limits_deg=None, spring=0.0, damping=0.0, target_deg=0.0,
          max_torque=1.0e6) -> Joint:
    """Register a revolute joint. ``spring`` in N m per degree, ``damping`` in N m per degree/s."""
    return _add_joint(Joint(name, "hinge", body0, body1, tuple(anchor), axis, limits_deg, spring, damping,
                            target_deg, max_torque))


def slider(name, body0, body1, anchor, axis="X", limits=None, stiffness=0.0, damping=0.0, target=0.0,
           max_force=1.0e3) -> Joint:
    """Register a prismatic joint. ``stiffness`` in N/m, ``damping`` in N s/m."""
    return _add_joint(Joint(name, "slider", body0, body1, tuple(anchor), axis, limits, stiffness, damping, target,
                            max_force))


def _drive_kind(j: Joint) -> str:
    return "angular" if j.kind == "hinge" else "linear"


def _author_joints(stage, prim_path: str, spec: Body) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for j in _JOINTS:
        if j.body1 != spec.name:
            continue
        b0 = _SPECS[j.body0]
        cls = UsdPhysics.RevoluteJoint if j.kind == "hinge" else UsdPhysics.PrismaticJoint
        path = f"{prim_path}/{j.name}"
        joint = cls.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([f"{parent}/{j.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        anchor = np.asarray(j.anchor, float)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(anchor - np.asarray(b0.pos, float)).tolist()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(anchor - np.asarray(spec.pos, float)).tolist()))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr(j.axis)
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        if j.limits is not None:
            joint.CreateLowerLimitAttr(float(j.limits[0]))
            joint.CreateUpperLimitAttr(float(j.limits[1]))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), _drive_kind(j))
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(j.stiffness))
        drive.CreateDampingAttr(float(j.damping))
        drive.CreateTargetPositionAttr(float(j.target))
        drive.CreateMaxForceAttr(float(j.max_force))
        _PATHS[j.name] = path
        _DRIVES.pop(j.name, None)


def set_drive(name: str, target: float | None = None, stiffness: float | None = None,
              damping: float | None = None) -> None:
    """Retune a joint's drive while the episode runs (hinge: degrees; slider: m). Every reset restores it.

    PhysX does not pick up a drive edited in USD mid-episode, so the joint is then switched off and on again, which
    has it rebuilt from USD with the new drive (its frames are unchanged, so its position still counts from the start).
    """
    if name not in _DRIVES:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        j = next(j for j in _JOINTS if j.name == name)
        prim = get_current_stage().GetPrimAtPath(_PATHS[name])
        drive = UsdPhysics.DriveAPI(prim, _drive_kind(j))
        joint = UsdPhysics.Joint(prim)
        enabled = joint.GetJointEnabledAttr() or joint.CreateJointEnabledAttr(True)
        _DRIVES[name] = (drive.GetTargetPositionAttr(), drive.GetStiffnessAttr(), drive.GetDampingAttr(), enabled)
    *attrs, enabled = _DRIVES[name]
    for attr, value in zip(attrs, (target, stiffness, damping)):
        if value is not None:
            attr.Set(float(value))
    enabled.Set(False)
    enabled.Set(True)


def _drives_reset() -> None:
    for j in _JOINTS:
        if j.name in _PATHS:
            set_drive(j.name, j.target, j.stiffness, j.damping)


def jointed() -> set[str]:
    return {n for j in _JOINTS for n in (j.body0, j.body1)}


# -- spawning (needs Isaac Sim) -----------------------------------------------------------------------------------------


def _spawn_body(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    material.func(phys, material)
    looks: dict = {}

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            surface = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            surface.func(path, surface)
            looks[color] = path
        return looks[color]

    body_path = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body_path)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    prim = xform.GetPrim()
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    if spec.kinematic:
        rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec.mass))
    if spec.ignore:
        parent = prim_path.rsplit("/", 1)[0]
        pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        for other in spec.ignore:
            pairs.AddTarget(f"{parent}/{other}/body")
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    rb.CreateSolverPositionIterationCountAttr(int(spec.iterations[0]))
    rb.CreateSolverVelocityIterationCountAttr(int(spec.iterations[1]))

    for i, part in enumerate(spec.parts):
        if isinstance(part, BoxPart):
            geom = UsdGeom.Cube.Define(stage, f"{body_path}/p{i}")
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*part.centre))
            if part.yaw:
                geom.AddOrientOp().Set(Gf.Quatf(math.cos(part.yaw / 2), 0.0, 0.0, math.sin(part.yaw / 2)))
            geom.AddScaleOp().Set(Gf.Vec3f(*part.size))
        else:
            geom = UsdGeom.Cylinder.Define(stage, f"{body_path}/p{i}")
            geom.CreateAxisAttr("Z")
            geom.CreateRadiusAttr(part.radius)
            geom.CreateHeightAttr(part.height)
            geom.AddTranslateOp().Set(Gf.Vec3d(*part.centre))
        path = str(geom.GetPrim().GetPath())
        bind_visual_material(path, look(tuple(part.color)), stage=stage)
        if part.collide:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
            collision.CreateContactOffsetAttr(0.002)
            collision.CreateRestOffsetAttr(0.0)
            bind_physics_material(path, phys, stage=stage)
    _author_joints(stage, prim_path, spec)
    return root.GetPrim()


def _arena_object(spec: Body):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class BodySpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_body)
        body_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=BodySpawnerCfg(body_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos),
                                    rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))


def as_numpy(x) -> np.ndarray:
    """An Isaac Lab array (a torch tensor, or a proxy with ``.torch``) as float64 NumPy."""
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def pose(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A Body's origin position (env frame, m) and orientation (xyzw)."""
    base = env.unwrapped
    p = as_numpy(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - as_numpy(base.scene.env_origins[0]), p[3:7]


def _state(env, spec: Body) -> ObjectState:
    data = env.unwrapped.scene[spec.name].data
    pos, quat = pose(env, spec.name)
    vel = as_numpy(data.root_com_vel_w if hasattr(data, "root_com_vel_w") else data.root_vel_w)[0]
    return body_state(spec, pos, quat, vel[:3], vel[3:6])


# -- hooks ------------------------------------------------------------------------------------------------------------

_RESET: list = []
_STEP: list = []


def _add_hook(hooks: list, fn):
    """Register ``fn``, replacing an earlier import's copy of the same function."""
    key = (fn.__module__, fn.__qualname__)
    hooks[:] = [h for h in hooks if (h.__module__, h.__qualname__) != key]
    hooks.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn()`` after every episode reset."""
    return _add_hook(_RESET, fn)


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    return _add_hook(_STEP, fn)


def _patch_scene(scene) -> None:
    if getattr(scene, "_jointkit", False):
        return
    plain_object, plain_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Body):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return plain_object(spec)

    def object_states(env, names):
        out = plain_states(env, [n for n in names if n not in _SPECS])
        out.update({n: _state(env, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object, object_states
    scene._jointkit = True


def _patch_embodiment(module) -> None:
    cls = getattr(module, "YamArenaEmbodiment", None)
    if cls is None or getattr(cls, "_jointkit", False):
        return
    plain_reset, plain_step, plain_sim = cls.reset, cls._task_step, cls._ensure_sim

    def _ensure_sim(self, *args, **kwargs):
        out = plain_sim(self, *args, **kwargs)
        held = getattr(self, "_free_objects", None)
        if held is not None:  # jittered one by one, jointed bodies would be pulled apart
            self._free_objects = [n for n in held if n not in jointed()]
        return out

    def reset(self, *args, **kwargs):
        _drives_reset()
        obs = plain_reset(self, *args, **kwargs)
        for fn in _RESET:
            fn()
        return obs

    def _task_step(self, *args, **kwargs):
        for fn in _STEP:
            fn(self._env)
        return plain_step(self, *args, **kwargs)

    cls.reset, cls._task_step, cls._ensure_sim = reset, _task_step, _ensure_sim
    cls._jointkit = True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Finder(importlib.abc.MetaPathFinder):
    """Patches each target module right after it is first imported."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        plain_exec = spec.loader.exec_module

        def exec_module(module):
            plain_exec(module)
            _TARGETS[fullname](module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena.scene and the embodiment, now if already imported, else when they load. Idempotent."""
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
