"""Checks of the mortar rule on hand-made trowel poses (no simulator needed):

    python -B -I tasks/replace-damaged-brick/test_mortar_rule.py

Mortar is laid only where the loaded blade, tilted 5 to 35 degrees, has its lower edge touching the pocket's floor
and sliding across it. A level blade, one held off the floor, an edge that doesn't move, or an empty trowel lays
nothing.
"""

import math
import pathlib

import numpy as np
from yam_arena import task_file

m = task_file.load(str(pathlib.Path(__file__).with_name("replace_brick.py"))).module


def about_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# The trowel along x, its blade in the pocket as far as the demo puts it.
BLADE_CENTRE_X = m.X_FACE + m.UNIT[1] - 0.006 - m.BLADE_SIZE[0] / 2
ORIGIN_X = BLADE_CENTRE_X - (m.BLADE_X[0] + m.BLADE_X[1]) / 2


def stroke(tilt_deg: float, gap: float, y0: float, y1: float, load: int = m.LOAD_CELLS,
           ledger: "m.MortarLedger | None" = None, steps: int = 70) -> "m.MortarLedger":
    """Move the trowel's handle axis along y from y0 to y1 in ``steps`` control steps (50 ms each), the blade
    turned ``tilt_deg`` about the trowel's length, its lowest edge ``gap`` metres above the floor."""
    ledger = ledger or m.MortarLedger(load=load)
    rot = about_x(math.radians(tilt_deg))
    for i in range(steps + 1):
        pos = np.array([ORIGIN_X, y0 + (y1 - y0) * i / steps, 0.0])
        _, edge = m.lowest_edge(pos, rot)
        pos[2] = m.BED + gap - edge[:, 2].min()
        ledger.apply(pos, rot, brick_in=False)
    return ledger


def axis_for_edge(tilt_deg: float, edge_y: float) -> float:
    """The handle axis's y that puts the middle of the tilted blade's lower edge at ``edge_y``."""
    _, edge = m.lowest_edge(np.array([ORIGIN_X, 0.0, 0.0]), about_x(math.radians(tilt_deg)))
    return edge_y - float(edge[:, 1].mean())


def covered(ledger: "m.MortarLedger") -> int:
    return int(ledger.bed.sum())


def main() -> None:
    nothing = {
        "a level blade 3 mm above the floor, swept end to end": stroke(0, 0.003, -0.068, 0.068),
        "a level blade lying on the floor, swept end to end": stroke(0, 0.0, -0.068, 0.068),
        "a blade tilted 3 degrees (too flat), edge on the floor, swept": stroke(3, 0.0, -0.075, 0.06),
        "a blade tilted 40 degrees (too steep), edge on the floor, swept": stroke(40, 0.0, -0.06, 0.06),
        "a tilted blade with its edge 3 mm above the floor, swept": stroke(10, 0.003, -0.075, 0.06),
        "a tilted blade with its edge on the floor, not moving": stroke(10, 0.0, -0.075, -0.075),
        "an empty trowel, tilted, edge on the floor, swept": stroke(10, 0.0, -0.075, 0.06, load=0),
    }
    for name, ledger in nothing.items():
        assert covered(ledger) == 0, f"{name}: {covered(ledger)} cells"
        print(f"ok  0 cells: {name}")

    # As the demo does: the lower edge from one end of the pocket to a blade's width short of the other.
    end = m.POCKET_HALF - 0.001
    stop = m.POCKET_HALF - 0.004 - m.BLADE_SIZE[1] * math.cos(math.radians(10))
    first = stroke(10, 0.0, axis_for_edge(10, -end), axis_for_edge(10, stop))  # -y edge down, from the right end
    assert 100 < covered(first) < m.N_BED, covered(first)
    print(f"ok  {covered(first)} cells: one stroke from the right end, edge down, toward the left")
    both = stroke(-10, 0.0, axis_for_edge(-10, end), axis_for_edge(-10, -stop), ledger=first)  # +y edge down, same load
    assert covered(both) == m.N_BED, covered(both)
    print(f"ok  {covered(both)} cells: a second stroke from the left end covers the whole floor "
          f"({m.LOAD_CELLS - both.load} of the load's {m.LOAD_CELLS} cells used)")


if __name__ == "__main__":
    main()
