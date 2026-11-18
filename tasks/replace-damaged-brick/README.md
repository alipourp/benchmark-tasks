# Replace a damaged brick in a wall, bedded in mortar

One task file, `replace_brick.py`, for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) on Dream Labs' station scene.
A damaged brick has been cut out of a brick wall, leaving a pocket with two broken pieces of it on the floor.
The robot clears the pieces into a bucket, spreads a bed of mortar over the pocket's floor with a trowel that
starts loaded, and sets the brick that matches the wall (not the longer Norman brick beside it) on the bed,
flush with the wall face.

## Requirements

- `yam-isaac-lab-arena` at commit `0059d02a1a91bc6834c1d0846022891df0c78730`, installed with its `setup.sh`
  (Isaac Lab-Arena `release/0.3.1` at `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`; `inspect-robots` 1.0.0,
  `inspect-robots-yam` 0.37.0 and `inspect-robots-dream` at `b5e936a201cec66c54c297e522c920d98445e339`).
  Tested on Isaac Sim 6.1 as pinned there.
- `source $ARENA_DIR/yam.env` in the shell.
- No third-party assets: every object is a primitive defined in the task file.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/replace-damaged-brick/replace_brick.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Replace the damaged brick: clear the broken pieces out of the pocket into the bucket, spread the mortar on the trowel over the pocket's bed, then set the brick that matches the wall on the bed, flush with the wall face." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/brick-dream
```

The scripted demo (full score): the same command with `--policy yam_scripted` and
`--log-dir ~/yam-eval/brick-demo`.

Videos: `for log in ~/yam-eval/brick-*/*.json; do inspect-robots video "$log"; done`.

If the machine has a desktop display that Isaac Sim can't open an OpenGL context on (`GLXBadFBConfig` at
startup), run with `DISPLAY` unset; the runs are headless.

## Score

`--scorer task_score` gives 1.0 when the task succeeds (a full score held for 1 s) and otherwise the task
file's `score()` at the end of the trial. Every step also reports `info["score"]`. The rules are in
`task.json` and in the file's `score_state()`; in short:

- **Broken pieces:** 0.15 for each piece in the bucket.
- **Mortar bed** (up to 0.3): the pocket's floor is a grid of 9 x 21 cells of about 1 cm. Mortar is laid the
  way a mason spreads it: the loaded blade tilted 5 to 35 degrees from level, its lower edge pressed on the floor
  and dragged across it. A cell gets mortar when a part of that edge touching the floor (within 1 mm) slides over
  it at 5 mm/s or more. A level blade, a blade held off the floor, or an edge pressed but not moving lays nothing.
  Only cells covered before the brick is 5 mm into the pocket count. 98% or more of the 189 cells gives 0.3, 95%
  0.24, 85% 0.15, 60% 0.075, else 0.
- **Brick** (up to 0.4): 0.4 when the replacement brick is seated (face within 3 mm of the wall face, bottom
  within 3 mm of the floor, centred within 5 mm, turned and tilted 3 degrees or less); 0.2 when it rests on the
  floor at least half inside the pocket, within 10 mm of centre and turned less than 10 degrees.
- **Hold:** a full score reads 0.9 until it has held for 1 s (20 control steps).

Checks of the scoring on fixed states (no simulator needed): everything done gives 1.0; everything but the
mortar gives 0.7; half the bed mortared gives 0.7 (below the 60% step). `test_mortar_rule.py` runs the mortar
rule on hand-made trowel poses: a level blade above or on the floor, a blade too flat or too steep, an edge held
off the floor, an edge that doesn't move, and an empty trowel all lay nothing; two tilted strokes cover the floor.

**Scoring a real trial:** mark the centres of the 9 x 21 cells (189 dots about 1 cm apart) on the pocket's floor
in black marker. Mortar is checked before the brick goes in, from the recording of the wrist camera on the hand
that spreads it, which looks into the pocket as the hand works: for each dot, take the last frame before the brick
reaches the wall face in which its spot is in view (not blocked by the trowel or a hand); the cell is covered if
mortar hides the dot there. Coverage is covered cells / 189; a cell whose spot is never in view counts as bare.
Real mortar has thickness, so the brick's
bed condition becomes a 10 +/- 3 mm gap to the brick above at both ends. Measure that, the head joints and the
face (flush within 3 mm at both ends) with a ruler and a straightedge across the wall face.

## Runtime and seeds

- Run limit: 2,400 control steps at 20 Hz (2 minutes). The scripted demo clears the two pieces one at a time
  with the right arm, spreads the bed in two tilted strokes (one from each end of the pocket, the trailing edge
  pressed on the floor), puts the trowel back on its board,
  then sets the brick with both arms and eases it home. It reaches 1.0 after about 104 s of simulated time and
  ends as a success about 1 s later (2,100 of the 2,400 steps).
- On an RTX 5080, one trial took about 4.5 minutes of wall-clock time, startup included.
- No randomness: the layout is fixed (`-E object_jitter=0`), so no seed is needed.

## How the file works

- **The trowel.** The task-file API has boxes and upright cylinders but no compound shapes, so the trowel (a stock
  5 x 2 in margin trowel) is one task object, its wooden handle (a box). Right after the official `yam_arena.scene` is imported, the file
  wraps its `task_object()` for that object name: after spawning the handle, it adds the steel shank and blade
  as more colliders of the same rigid body. The hook is installed only where Isaac Lab can be imported, it does
  nothing to other objects, and a second load of the file is served by the first wrapper. Keep
  `-E object_jitter=0`.
- **Mortar ledger.** `score()` is called after every control step with each object's state. The file keeps the
  floor's 9 x 21 cells. Each step it finds the blade's lowest underside edge from the trowel's pose and checks it
  at points 3 mm apart: a point that touches the floor (within 1 mm above it, or up to 4 mm into it, as the
  simulator lets a pressed edge sit) on this step and the last, with the blade tilted 5 to 35 degrees, and that
  has moved at least 0.25 mm (5 mm/s), marks the cells within 3 mm of its path. In the demo, the edge is 0 to 0.7 mm into the floor on every step
  that lays mortar (the hand is commanded 2 mm lower): the blade is pressed on the floor, not held above it. The simulator reports an
  object's position as its bounding box's centre, so the file recovers the handle's centre from it first. The
  trowel starts with a load of 240 cells; pressing the blade flat (within 10 degrees) into the board's mortar
  patch reloads it. Once the replacement brick is 5 mm into the pocket, the ledger freezes.
  `success()` and `score()` share each step's state dict, so a step counts once.
- **Episodes.** Task files get no reset hook. The file finds the calling embodiment (the caller with an
  episode clock, a task and a `reset`) and wraps its `reset` (keeping its name and signature, and never twice),
  so every new trial starts a fresh ledger and hides the last trial's tiles before its first frame renders. If
  no such caller is found, it warns once and treats the trowel resting at its start pose as a new trial.
- **Visible mortar.** Mortar on the floor, and the load on the blade, are drawn as thin tiles with Isaac Lab's
  `VisualizationMarkers`, imported inside `score()` once the simulator is running (the module itself imports
  nothing from Isaac Sim, as task files must), so camera-based policies see what they have spread. The tiles
  are only a picture of the ledger: if drawing fails, they turn off with one warning and scoring carries on.
- **Side grips in the demo.** `yam_arena`'s scripted player only grips straight down. The demo needs side
  grips (the brick goes in under the course above), so a waypoint's yaw may be a `SideGrip`, a float that also
  carries the grip's orientation, and the file wraps `yam_arena.demo.grip_rotation` to honour it; any other
  waypoint is unchanged.
- **The demo's tuning.** The arms sag under the brick's weight, more the further they reach, so the demo
  raises its hands by a reach-dependent allowance while carrying, rises a further 22 mm while pushing the brick
  into the pocket, takes the allowance away once the bed carries the brick, and eases it home in three
  loosen-slide-grip steps. Holding the trowel by its handle's sides, the hand is pitched up 2 degrees and turned
  6 degrees about the handle to hold the blade level; to spread, it turns the blade a further 10 degrees each way
  and presses its lower edge 2 mm into the floor (commanded) while it draws it along the pocket.
- **No `from __future__ import annotations`.** The task loader imports the file without registering it in
  `sys.modules`, and dataclasses then cannot resolve string annotations.

## Checking it without a GPU

The file loads without Isaac Sim, so `task_file.load()` checks its objects and hooks on any machine with
`yam-isaac-lab-arena` installed:

```bash
python -B -I -c "from yam_arena import task_file; t = task_file.load('tasks/replace-damaged-brick/replace_brick.py'); print(t.names)"
python -B -I tasks/replace-damaged-brick/test_mortar_rule.py
```
