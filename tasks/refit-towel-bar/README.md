# Refit a spring-loaded towel bar

## Run a trial

```bash
inspect-robots run --embodiment yam_arena \
  -E task=mount_towel_bar.py -E object_jitter=0 \
  --instruction "Mount the towel bar between the two wall brackets." \
  --scorer task_score --max-steps 2400 --no-prompt --store-frames
```

Runtime: approximately 50–55 seconds of simulated time per trial (1,000–1,100 steps
at 20 Hz). The run limit is 2,400 steps (2 minutes).

No seeds; the layout is fixed (`object_jitter=0`).

## Scripted demo

The closed-loop demo in `mount_towel_bar.py` (`demo_script`) uses `cabledemo.py`:

```bash
python run_demo.py mount_towel_bar.py out/ --max-steps 2400
```

The demo scores 1.0 in approximately 1,055 steps. It reads bar and post poses from
the simulator for yaw correction but does not teleport or edit any object.

### Demo procedure

1. Both arms move to a wide hover 40 mm from their rest y, then slide inward to
   grip hover at y = ±100 mm. A convergence loop repeats the command until both TCPs
   are within 30 mm of their targets.
2. Both arms descend and grip the bar (left at y = +100 mm, right at y = −100 mm).
3. Lift the bar to bracket height (10.0 cm above the table) and move forward to the
   channel centre x (23.2 cm).
4. Yaw correction: if the bar yaw exceeds 5 degrees, the arms nudge toward the wall.
5. Push left by 60 mm — this compresses the spring post against the left bracket.
6. Slide right by 84 mm in two stages — the right end enters the right channel and the
   spring post extends into the left channel.
7. Release and move both arms clear.

## Scoring

| Points | Condition | Physical check |
|--------|-----------|----------------|
| 0.25 | Spring compressed while bar aligned | A steel rule shows the post has retracted at least 20 mm; both bar ends are within 4 cm of their bracket openings. Captured from the top-down camera. |
| 0.25 | Right end seated | The right bar end is at least 6 mm inside the right channel. Measure with a depth gauge from the channel opening. |
| 0.25 | Both ends seated | Both ends are at least 6 mm inside their channels and the bar is level within 5 degrees. A spirit level on the bar and depth checks on both sides. |
| 0.25 | Stable and clear | The bar remains seated for 1 s with hands at least 4 cm clear. Bar speed under 3 mm/s, rotation under 3 deg/s. In simulation, the scorer tracks velocity over 20 consecutive steps. |

Partial credit is cumulative. Each condition, once met, is latched (it stays true even
if the bar shifts afterward), except the final stability condition, which requires 1
continuous second of both-seated stability.

## What is simplified

- The spring rate (250 N/m) is an assumed value, not measured from a commercial
  product. Real spring-loaded towel bars vary from roughly 100–400 N/m.
- The bar has a square cross-section. Real bars are typically round or oval, but
  the cross-section shape does not affect the compress-and-slide mechanism.
- The bracket channels are plain rectangular pockets. Real brackets may have chamfered
  or rounded entry edges that make alignment slightly easier.
- The wall panel is kinematic (infinitely rigid and fixed). Real walls flex slightly
  under load but this does not affect the task.
- Contact friction is uniform (0.5 on bar and post, 0.3 on brackets). Real chrome
  surfaces vary with wear and cleaning.
- Solver iterations: 64 position, 8 velocity for jointed bodies.

## Parts for real trials

| Part | Description | Approx. cost |
|------|-------------|-------------|
| Wall panel | Plywood or MDF board, 35 × 25 × 2 cm, painted white, with two recessed cup brackets mounted 24 cm apart (centre to centre), 10 cm above the base | $15 |
| Bracket pair | Chrome or stainless steel recessed cup brackets, channel opening 50 × 50 mm, 55 mm deep; or equal | $10–20 |
| Towel bar | Chrome spring-loaded towel rod, 22 cm bar with 1.8 cm square cross-section, internal spring post with ~40 mm travel; or a 10-inch spring-loaded closet rod trimmed to fit, or equal | $8–15 |
| Foot / base | A 35 × 7 × 2 cm wood foot screwed to the back of the panel to keep it upright on the table | $3 |

Total: approximately $36–53 per setup. Nothing is consumed between trials.

## Files

- `mount_towel_bar.py` — task definition: scene objects, joints, scoring, demo script
- `jointkit.py` — rigid-body joint library (shared)
- `cabledemo.py` — closed-loop demo framework (shared)
- `run_demo.py` — launcher that pre-imports `inspect_robots_yam`
- `check.py` — offline scorer verification (6 scripted variants, no simulator needed)
- `task.json` — submission metadata
