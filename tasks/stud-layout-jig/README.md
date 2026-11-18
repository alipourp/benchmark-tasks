# Frame a short wall section: set three studs on their layout marks and screw each through the bottom plate

One task file, `stud_layout_jig.py`, for the `yam_arena` embodiment of `yam-isaac-lab-arena` on Dream Labs' station
scene, with its screw model `screw_runtime.py` and the shared compound-part helper `yam_parts.py`.
A short wall section is framed flat on a framing table, as carpenters frame walls on the deck: two 2x4 plates stand
on edge a stud's length plus 6 mm apart, each marked for three studs 16 in on centre.
The robot sets each 12 in stud on edge between the plates on its marks, then takes up a cordless impact driver and
drives the screw already started through the bottom plate at that mark into the stud's end, drawing the stud tight
to the plate, until the screw's head is flush.

## Requirements

- `yam-isaac-lab-arena` installed with its `setup.sh` (Isaac Lab-Arena, `inspect-robots` and `inspect-robots-yam`),
  or the Robocurve Brev launchable; a Dream API key in `~/.dreamlabs/credentials` for the Dream-1 run.
- No third-party assets: every object is a primitive (boxes, cylinders and spheres) defined in the task files.

## Run a trial

From this folder, with 20 Hz control:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=stud_layout_jig.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Frame the wall section: stand each stud on edge between the plates on its layout marks, then drive the screw at its mark through the bottom plate into the stud's end with an impact driver until the head is flush." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/stud-wall-dream

# The scripted demo (full score): the same command with
#   --policy yam_scripted --log-dir ~/yam-eval/stud-wall-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/stud-wall-*/*.json; do inspect-robots video "$log"; done
```

If the machine has a desktop display that Isaac Sim can't open an OpenGL context on (`GLXBadFBConfig` at startup),
run with `DISPLAY` unset (`env -u DISPLAY inspect-robots run ...`); the runs are headless.

## Score

`--scorer task_score` gives 1.0 when the task succeeds (all three studs fastened, both hands at least 3 cm from every
stud, held for 1 s) and otherwise the task file's `score()` at the end of the trial. The rules are in `task.json`
and in the file's scoring section; in short, each stud scores up to 1/3 on its own and each mark credits at most one
stud:

| Credit | Condition | Check on a real framing table |
|---|---|---|
| 0.10 placed | on edge between the plates (upright within 3 degrees), square to the plates within 3 degrees, bottom within 3 mm of the plywood, centre line within 4 mm of the centre of its marks | tape measure across the marks; a speed square |
| 0.10 caught | its screw has gone into the stud's end | the stud can no longer be lifted out |
| 0.133 fastened | the screw driven home, its head within 1 mm of flush with the plate, and the stud's end tight to the bottom plate | a straightedge across the plate's face, or a 1 mm feeler gauge that will not go under the head's edge; a 1 mm feeler gauge does not go into the joint |

A screw driven with no stud on its marks catches nothing. A stud anywhere else (lying flat, on the stops' tops, off
its marks, tilted, turned or on the table) scores 0. In simulation the trial is voided (scores 0) if a stud is
missing, a pose is not finite, a rotation is not unit length, or a stud's reported outline doesn't match a
38 x 89 x 305 mm box.

The scoring, the screw model and the demo's reach can be checked without the simulator (40 checks):

```bash
python -m pytest -q test_stud_layout_jig.py
```

## Runtime and seeds

- Run limit: 2,400 control steps at 20 Hz (2 minutes).
- The scripted demo (`demo()`): the left arm sets the left stud, the right arm the middle stud and then the right
  one; then each arm takes up its impact driver, drives the screws at its studs (the left arm one, the right arm two),
  pressing each home, and lifts the driver clear. The plans last 53 s (left arm) and 100 s (right arm); the trial
  reached 1.0 and ended as a success at step 1,905 (95 s of simulated time). On an RTX 5080 a trial took about
  10 minutes of wall-clock time, startup included.
- No randomness: the layout is fixed (`-E object_jitter=0`), so no seed is needed.

## How the files work

- **The screws** (`screw_runtime.py`). PhysX cannot cut a thread into wood, so each screw is a kinematic body the
  model holds on its axis and moves only as it is driven; its head is solid, so it stops the driver's bit as a
  screw held by its thread does. A screw turns in, at up to 25 mm/s, only while all of these hold, each read from
  PhysX contacts at every physics substep: the bit's tip touches the head (a contact within 3 mm of the tip, the tip
  within 3 mm of the screw's axis, the bit within 8 degrees of it); the tip presses the head along the screw's axis
  with at least **20 N**; and a pad presses that driver's trigger with at least **17 N**. As the screw goes in its
  head draws away from the bit and the push falls, so the hand has to follow the head in, pressing, for the screw to
  keep turning. As the screw's tip leaves the plate it catches the stud whose end is there (square to the plate, the
  screw at least 6 mm inside the end's edges) and draws that end tight to the plate. A screw driven with no stud
  there holds nothing. The model runs before every physics substep and applies its forces through Isaac Lab, after
  the screw model of the repository's lever-set task (`tasks/fit-lever-set-to-bored-door/lockset_runtime.py`); the
  contacts come from a PhysX rigid-contact view, as in `tasks/move-live-wire-with-stick` (contact reporting is
  switched on for the two drivers when they are spawned).
- **Why 20 N.** Screwdriving studies with drills and screwdrivers apply 44-100 N of feed force (particleboard,
  solid softwood). An impact driver needs less: its hammer turns the screw in short blows, so a square-drive bit
  does not cam out under a lighter push, and this screw is already started through a clearance hole, its tip on the
  stud's end. 20 N is also near what a YAM arm can push along the screw while its hand squeezes the trigger (the
  wrist joints, 7 N m, take most of the load): at 25 N the right arm uses up to 97 % of a wrist joint's limit at the
  outer screw.
- **The trigger.** Each driver has a trigger, a red block standing 6 mm proud of the grip's front face (40 x 28 mm),
  at the grip's balance point, where the hand closes on it. The drivers are held by their grips from above, one pad
  on the trigger's face and the other on the grip's back. The trigger is pulled only while PhysX reports the robot
  pressing its face with at least 17 N along its travel (a power tool's trigger takes about 10-20 N; its travel is
  not simulated). A hold (pads driven 2.8 mm past the faces, about 14 N) carries a driver without pulling it; a
  squeeze (8 mm, about 40 N) pulls it. Pushing the bit into a head with F takes F/2 off the trigger pad and adds it
  to the back pad, so with the squeeze the trigger stays pulled for pushes up to about 33 N. In a real trial the
  pad behind the grip carries the trigger's 17 N plus the 20 N push, 37 N, within the YAM's roughly 40 N per finger.
  After a screw is home the hand eases the push off before relaxing to the hold, so the pads never go slack on the
  grip.
- **Pushing in the demo.** Each arm joint is driven as a spring (stiffness 80 N m/rad at the shoulder, 10 at the
  wrist), so to press the bit into a head with 23 N the demo commands the hand ahead of where the head stops it,
  by K^-1 J^T W in the joints (`_pressing`), and follows the head home in one steady move while the screw turns.
- **Compound parts.** The task-file API has boxes, spheres and upright cylinders, so the plates (with their pilot
  holes), screws, drivers and rests are `yam_parts.Part`s, from the repository's shared helper, copied unchanged.
- **Reads in the demo.** Carried in one hand, a stud sags and tips in the grip by more than the 3 mm the plates and
  stops leave, and a driver sags by more than its bit can miss a screw's head by. So the demo reads the stud's or the
  driver's pose at fixed points and moves the hand to correct it, as scripted demos may: over each slot, above the
  plates, just above the stops and between the plates before letting go of a stud; and before driving each screw.
- **Tilted grips.** With the hand straight down the arms cannot lift a stud clear of the plates, so the demo tips the
  fingertips 30 degrees along the stud; `yam_arena.demo.grip_rotation` is wrapped to take a full rotation for such
  waypoints, and every other waypoint is unchanged.
