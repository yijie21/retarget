"""Build a MuJoCo scene with two UR3e arms side by side.

Uses MjSpec.attach to namespace each arm (left_/right_) so joint, body,
and actuator names don't collide. Writes the compiled model to scene.xml.
"""

import math
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
UR3E_DIR = ASSETS / "universal_robots_ur3e"
UR3E_XML = UR3E_DIR / "ur3e.xml"
UR3E_ASSETS = UR3E_DIR / "assets"

SHARPA_DIR = (
    ASSETS / "sharpa_assets" / "SharpaWave_URDF_XML_USD_V3.0.0" / "src"
)
LEFT_HAND_XML = SHARPA_DIR / "left_sharpa_wave" / "left_sharpa_wave.xml"
RIGHT_HAND_XML = SHARPA_DIR / "right_sharpa_wave" / "right_sharpa_wave.xml"

# Side-by-side layout: both arms facing +x, separated along y.
# 26 inches base-to-base = 0.6604 m total, so ±0.3302 m from origin.
INCH = 0.0254
SEPARATION = 26 * INCH

# Table: 4 ft × 4 ft slab on 4 legs, top face 25 in above the floor. The slab
# is the same thickness as the breadboard so the two are coplanar (aligned
# bottoms and tops) — the breadboard sits flush in the tabletop, not on top.
FOOT = 12 * INCH
TABLE_HEIGHT = 25 * INCH
SLAB_THICKNESS = 0.5 * INCH
TABLE_HALF = (4 * FOOT / 2, 4 * FOOT / 2, SLAB_THICKNESS / 2)
TABLE_TOP_Z = TABLE_HEIGHT
SLAB_CENTER_Z = TABLE_TOP_Z - TABLE_HALF[2]

# 4 legs (2 in square cross-section), inset 3 in from each tabletop edge.
LEG_HALF_XY = 1 * INCH
LEG_HEIGHT = TABLE_TOP_Z - SLAB_THICKNESS  # floor to underside of slab
LEG_HALF = (LEG_HALF_XY, LEG_HALF_XY, LEG_HEIGHT / 2)
LEG_INSET = 3 * INCH
LEG_OFFSET = TABLE_HALF[0] - LEG_INSET - LEG_HALF_XY

# Corner radius for the rounded tabletop.
CORNER_R = 2 * INCH

# Optical breadboard sitting ON TOP of the tabletop, with its -x (near) edge
# flush with the table's -x edge. Lifted 1 in off the tabletop on wooden
# support blocks at the 4 corners.
BREADBOARD_HALF = (FOOT / 2, 3 * FOOT / 2, 0.5 * INCH / 2)
BREADBOARD_X = -TABLE_HALF[0] + BREADBOARD_HALF[0]  # near-edge alignment
# In y the breadboard is *not* centered: its -y edge sits 18.25 cm from the
# table's -y edge. The whole arm+block+support assembly rides with it.
BREADBOARD_RIGHT_GAP = 0.1825
BREADBOARD_Y = -TABLE_HALF[1] + BREADBOARD_RIGHT_GAP + BREADBOARD_HALF[1]
BREADBOARD_LIFT = 1 * INCH
BREADBOARD_BOTTOM_Z = TABLE_TOP_Z + BREADBOARD_LIFT
BREADBOARD_CENTER_Z = BREADBOARD_BOTTOM_Z + BREADBOARD_HALF[2]
BREADBOARD_TOP_Z = BREADBOARD_BOTTOM_Z + 2 * BREADBOARD_HALF[2]

# Wooden support blocks: 4 in × 4 in × 1 in, filling the gap under the
# breadboard's four corners.
SUPPORT_HALF = (2 * INCH, 2 * INCH, BREADBOARD_LIFT / 2)

# Arms ride with the breadboard; 0.75 in forward of its center in x
# (preserves the previous breadboard-to-arm offset).
ARM_X = BREADBOARD_X + 0.75 * INCH

# Mounting block under each arm: 18 × 18 × 1 cm, sitting on the breadboard.
BLOCK_HALF = (0.09, 0.09, 0.005)
BLOCK_TOP_Z = BREADBOARD_TOP_Z + 2 * BLOCK_HALF[2]

LEFT_POS = (ARM_X, BREADBOARD_Y + SEPARATION / 2, BLOCK_TOP_Z)
RIGHT_POS = (ARM_X, BREADBOARD_Y - SEPARATION / 2, BLOCK_TOP_Z)

LEFT_EULER_DEG = (0.0, 0.0, -68.27)
RIGHT_EULER_DEG = (0.0, 0.0, -111.73)

# Target joint angles (rad) — order matches the UR3e joint chain:
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
LEFT_QPOS = [
    -1.830804173146383,
    -1.49032814920459,
    -0.2515060305595398,
    -2.9926845035948695,
    -1.4880836645709437,
    3.9406208992004395,
]
RIGHT_QPOS = [
    2.0238232612609863,
    -1.56756230656568,
    0.104102913533346,
    -0.07167549551043706,
    1.6591987609863281,
    -3.932023588811056,
]


def build() -> mujoco.MjSpec:
    parent = mujoco.MjSpec()
    parent.modelname = "dual_ur3e"
    # Otherwise attach moves the child subtree, preventing a second attach.
    parent.copy_during_attach = True
    # Absolute meshdir so the saved scene.xml can find the UR3e .obj files
    # regardless of where it's loaded from.
    parent.compiler.meshdir = str(UR3E_ASSETS)

    # Floor + light.
    parent.worldbody.add_light(name="top", pos=[0, 0, 2], dir=[0, 0, -1])
    parent.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.1],
        rgba=[0.85, 0.85, 0.9, 1],
    )
    # Tabletop: rounded-corner slab, decomposed into non-overlapping pieces
    # (1 central box + 4 edge strips + 4 corner cylinders) to avoid
    # z-fighting and stacked shadows.
    a, b, h = TABLE_HALF
    top_rgba = [0.95, 0.95, 0.95, 1]
    parent.worldbody.add_geom(
        name="table_center",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[a - CORNER_R, b - CORNER_R, h],
        pos=[0, 0, SLAB_CENTER_Z],
        rgba=top_rgba,
    )
    # Edge strips (each covers one side of the inner box, between the two
    # adjacent corner cylinders).
    parent.worldbody.add_geom(
        name="table_edge_px",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CORNER_R, b - CORNER_R, h],
        pos=[a - CORNER_R, 0, SLAB_CENTER_Z],
        rgba=top_rgba,
    )
    parent.worldbody.add_geom(
        name="table_edge_nx",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CORNER_R, b - CORNER_R, h],
        pos=[-(a - CORNER_R), 0, SLAB_CENTER_Z],
        rgba=top_rgba,
    )
    parent.worldbody.add_geom(
        name="table_edge_py",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[a - CORNER_R, CORNER_R, h],
        pos=[0, b - CORNER_R, SLAB_CENTER_Z],
        rgba=top_rgba,
    )
    parent.worldbody.add_geom(
        name="table_edge_ny",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[a - CORNER_R, CORNER_R, h],
        pos=[0, -(b - CORNER_R), SLAB_CENTER_Z],
        rgba=top_rgba,
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            tag = f"{'n' if sx < 0 else 'p'}{'n' if sy < 0 else 'p'}"
            parent.worldbody.add_geom(
                name=f"table_corner_{tag}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[CORNER_R, h, 0],
                pos=[sx * (a - CORNER_R), sy * (b - CORNER_R), SLAB_CENTER_Z],
                rgba=top_rgba,
            )
    # 4 legs supporting the tabletop from underneath.
    for sx in (-1, 1):
        for sy in (-1, 1):
            parent.worldbody.add_geom(
                name=f"leg_{'n' if sx < 0 else 'p'}{'n' if sy < 0 else 'p'}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(LEG_HALF),
                pos=[sx * LEG_OFFSET, sy * LEG_OFFSET, LEG_HEIGHT / 2],
                rgba=[0.75, 0.75, 0.75, 1],
            )
    # Room walls. Back wall (white) stands 23.5 cm past the -x edge, behind
    # the arms. (Far concrete wall, 17 in past the +x edge, removed for now.)
    wall_thickness = 0.05  # half-thickness
    wall_half_h = 1.5
    wall_z = wall_half_h
    wall_half_span = 2.0
    far_wall_gap = 17 * INCH
    back_wall_gap = 0.235
    far_wall_x = TABLE_HALF[0] + far_wall_gap + wall_thickness
    back_wall_x = -(TABLE_HALF[0] + back_wall_gap + wall_thickness)
    # Far wall removed for now.
    # parent.worldbody.add_geom(
    #     name="far_wall",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=[wall_thickness, wall_half_span, wall_half_h],
    #     pos=[far_wall_x, 0, wall_z],
    #     rgba=[0.55, 0.55, 0.52, 1],
    # )
    parent.worldbody.add_geom(
        name="back_wall",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[wall_thickness, wall_half_span, wall_half_h],
        pos=[back_wall_x, 0, wall_z],
        rgba=[0.98, 0.98, 0.98, 1],
    )

    # Foam pad: 3 ft (x) × 4 ft (y) × 1.5 in, filling the tabletop area
    # behind the breadboard (between breadboard's +x edge and table's +x edge).
    foam_half = (3 * FOOT / 2, 4 * FOOT / 2, 1.5 * INCH / 2)
    breadboard_far_x = BREADBOARD_X + BREADBOARD_HALF[0]
    foam_x = breadboard_far_x + foam_half[0]
    parent.worldbody.add_geom(
        name="foam",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(foam_half),
        pos=[foam_x, 0, TABLE_TOP_Z + foam_half[2]],
        rgba=[0.88, 0.84, 0.72, 1],
    )

    # Wooden support blocks under the breadboard corners (flush inside).
    for sx in (-1, 1):
        for sy in (-1, 1):
            tag = f"{'n' if sx < 0 else 'p'}{'n' if sy < 0 else 'p'}"
            bx = BREADBOARD_X + sx * (BREADBOARD_HALF[0] - SUPPORT_HALF[0])
            by = BREADBOARD_Y + sy * (BREADBOARD_HALF[1] - SUPPORT_HALF[1])
            parent.worldbody.add_geom(
                name=f"support_{tag}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(SUPPORT_HALF),
                pos=[bx, by, TABLE_TOP_Z + SUPPORT_HALF[2]],
                rgba=[0.55, 0.4, 0.22, 1],
            )
    # Optical breadboard on top of the support blocks, -x edge flush with table -x edge.
    parent.worldbody.add_geom(
        name="breadboard",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BREADBOARD_HALF),
        pos=[BREADBOARD_X, BREADBOARD_Y, BREADBOARD_CENTER_Z],
        rgba=[0.12, 0.12, 0.12, 1],
    )

    # Load UR3e and attach twice at prefixed frames. Each frame also carries
    # a blue mounting block that shares the arm's xy position and yaw.
    arm = mujoco.MjSpec.from_file(str(UR3E_XML))
    for prefix, pos, euler in (
        ("left_", LEFT_POS, LEFT_EULER_DEG),
        ("right_", RIGHT_POS, RIGHT_EULER_DEG),
    ):
        block_pos = (pos[0], pos[1], BREADBOARD_TOP_Z + BLOCK_HALF[2])
        parent.worldbody.add_geom(
            name=f"{prefix}block",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(BLOCK_HALF),
            pos=list(block_pos),
            euler=list(euler),
            rgba=[32 / 255, 37 / 255, 43 / 255, 1],
        )
        frame = parent.worldbody.add_frame(pos=list(pos), euler=list(euler))
        parent.attach(arm, prefix=prefix, frame=frame)

    # Flange-to-hand coupler: a black base cylinder against the arm,
    # followed by a silver cylinder against the hand. Both oriented along
    # the wrist_3_joint axis (which is +Y in wrist_3_link's frame).
    SITE_POS_Y = 0.09215
    BLACK_R, BLACK_H = 0.034, 0.01325
    SILVER_R, SILVER_H = 0.021, 0.0175
    HAND_OFFSET = BLACK_H + SILVER_H  # 0.03075 m along site +Z
    HAND_RGBA = [0.79216, 0.81961, 0.93333, 1]
    SITE_QUAT = [-1, 1, 0, 0]  # same as the UR3e attachment_site

    for prefix in ("left", "right"):
        wrist = parent.body(f"{prefix}_wrist_3_link")
        wrist.add_geom(
            name=f"{prefix}_coupler_black",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[BLACK_R, BLACK_H / 2, 0],
            pos=[0, SITE_POS_Y + BLACK_H / 2, 0],
            quat=SITE_QUAT,
            rgba=[0.05, 0.05, 0.05, 1],
        )
        wrist.add_geom(
            name=f"{prefix}_coupler_silver",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[SILVER_R, SILVER_H / 2, 0],
            pos=[0, SITE_POS_Y + BLACK_H + SILVER_H / 2, 0],
            quat=SITE_QUAT,
            rgba=HAND_RGBA,
        )

    # Attach Sharpa Wave hands to each arm's flange (attachment_site).
    # The site's +Z axis aligns with the wrist_3_joint axis, so:
    #   - root pos shift along +Z offsets the hand past the coupler.
    #   - root quat about Z is the wrist roll (mount rotation).
    left_hand = mujoco.MjSpec.from_file(str(LEFT_HAND_XML))
    right_hand = mujoco.MjSpec.from_file(str(RIGHT_HAND_XML))

    def z_quat(deg):
        t = math.radians(deg)
        return [math.cos(t / 2), 0, 0, math.sin(t / 2)]

    left_hand.body("left_hand_C_MC").pos = [0, 0, HAND_OFFSET]
    left_hand.body("left_hand_C_MC").quat = z_quat(45)
    right_hand.body("right_hand_C_MC").pos = [0, 0, HAND_OFFSET]
    right_hand.body("right_hand_C_MC").quat = z_quat(135)

    parent.attach(
        left_hand, prefix="left_hand_", site=parent.site("left_attachment_site")
    )
    parent.attach(
        right_hand, prefix="right_hand_", site=parent.site("right_attachment_site")
    )

    # Keyframe: specified arm angles + zero for every hand joint.
    # qpos layout follows the body tree (left arm → left hand → right arm →
    # right hand), but actuators are grouped by spec attach order
    # (both arms, then both hands), so ctrl needs its own layout.
    hand_dof = 22  # 22 joints per Sharpa Wave hand
    home_qpos = (
        list(LEFT_QPOS) + [0.0] * hand_dof
        + list(RIGHT_QPOS) + [0.0] * hand_dof
    )
    home_ctrl = (
        list(LEFT_QPOS) + list(RIGHT_QPOS)
        + [0.0] * hand_dof + [0.0] * hand_dof
    )
    parent.add_key(name="home", qpos=home_qpos, ctrl=home_ctrl)

    return parent


def main() -> None:
    spec = build()
    model = spec.compile()
    print(f"Compiled OK — nbody={model.nbody}, nq={model.nq}, nu={model.nu}")

    out = HERE / "scene.xml"
    out.write_text(spec.to_xml())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
