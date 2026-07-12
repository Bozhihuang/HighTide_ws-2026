#!/usr/bin/env python3
"""
Generate professional presentation diagrams for the HighTide AUV software stack.

Pure-matplotlib (offline) renderer that emits clean, company-deck-style
node-and-arrow diagrams as high-resolution PNGs. Run:  python generate_diagrams.py
Outputs land next to this script.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- typography -------------------------------------------------------------
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["svg.fonttype"] = "none"

# ---- palette (light, professional) ------------------------------------------
INK      = "#10222f"
MUTED    = "#5c7484"
BORDER   = "#c9d7df"
EDGE     = "#8598a5"
SHADOW   = "#dfe7ec"
ACCENT   = "#d9821f"          # innovation highlight
PAGE     = "#ffffff"

C = {
    "sensor":  "#4d7d9b",
    "perc":    "#12866b",
    "mission": "#0d8399",
    "nav":     "#3a68bf",
    "control": "#b25f2c",
    "neutral": "#6b8393",
}
LABEL = {
    "sensor": "Sensing", "perc": "Perception", "mission": "Mission brain",
    "nav": "Navigation", "control": "Control / actuation",
}

R = 1.4  # corner rounding (data units)


def tint(c, w):
    r, g, b = mc.to_rgb(c)
    return (r + (1 - r) * w, g + (1 - g) * w, b + (1 - b) * w)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(PAGE)
    return fig, ax


def card(ax, x, y, w, h, title, sub="", cat="neutral", key=False,
         tsize=11, ssize=8.4, title_color=None):
    color = C.get(cat, C["neutral"])
    # soft shadow
    ax.add_patch(FancyBboxPatch((x + 0.6, y - 0.7), w, h,
                 boxstyle=f"round,pad=0,rounding_size={R}",
                 fc=SHADOW, ec="none", zorder=1))
    if key:  # amber glow behind
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.9), w + 1.8, h + 1.8,
                     boxstyle=f"round,pad=0,rounding_size={R+0.6}",
                     fc=tint(ACCENT, 0.72), ec="none", zorder=1.5, alpha=0.6))
    ec = ACCENT if key else color
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={R}",
                 fc=tint(color, 0.90), ec=ec, lw=2.0 if key else 1.4, zorder=2))
    # category dot
    ax.add_patch(Circle((x + 2.4, y + h - 2.6), 0.75, fc=color, ec="none", zorder=3))
    if key:
        ax.text(x + w - 1.4, y + h - 1.2, "★", ha="right", va="top",
                fontsize=10, color=ACCENT, zorder=4, fontweight="bold")
    ax.text(x + 4.2, y + h - 2.6, title, ha="left", va="center",
            fontsize=tsize, color=title_color or INK, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + 2.6, y + h - 5.4, sub, ha="left", va="top",
                fontsize=ssize, color=MUTED, zorder=4, family="monospace",
                linespacing=1.45)
    return dict(x=x, y=y, w=w, h=h)


def L(c):  return (c["x"], c["y"] + c["h"] / 2)
def Rt(c): return (c["x"] + c["w"], c["y"] + c["h"] / 2)
def T(c):  return (c["x"] + c["w"] / 2, c["y"] + c["h"])
def B(c):  return (c["x"] + c["w"] / 2, c["y"])


def arrow(ax, p0, p1, label="", rad=0.0, color=EDGE, lw=2.0, lsize=7.6):
    ax.add_patch(FancyArrowPatch(p0, p1,
                 arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1.2,
                 shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=lsize,
                family="monospace", color=MUTED, zorder=5,
                bbox=dict(boxstyle="round,pad=0.28", fc=PAGE, ec=BORDER, lw=0.8))


def header(ax, W, title, sub=""):
    ax.text(4, W["top"], title, ha="left", va="top", fontsize=17,
            fontweight="bold", color=INK)
    if sub:
        ax.text(4, W["top"] - 5.4, sub, ha="left", va="top", fontsize=9.6,
                color=MUTED)
    ax.text(W["right"], W["top"], "HighTide · RoboSub 2026", ha="right", va="top",
            fontsize=9, color=MUTED, family="monospace")
    ax.plot([4, W["right"]], [W["top"] - 8.6, W["top"] - 8.6],
            color=BORDER, lw=1.2, zorder=0)


def legend(ax, x, y, cats):
    """Single-row inline legend (compact vertical footprint)."""
    ax.text(x, y, "LEGEND", fontsize=7.5, color=MUTED, family="monospace",
            fontweight="bold", va="center")
    x0 = x + 17
    for k in cats:
        ax.add_patch(Circle((x0, y), 0.8, fc=C[k], ec="none"))
        ax.text(x0 + 1.8, y, LABEL[k], ha="left", va="center", fontsize=8.2, color=INK)
        x0 += 5.5 + len(LABEL[k]) * 1.05
    ax.text(x0, y, "★", ha="left", va="center", fontsize=10, color=ACCENT,
            fontweight="bold")
    ax.text(x0 + 2.2, y, "Key innovation", ha="left", va="center", fontsize=8.2,
            color=ACCENT)


def note(ax, x, y, w, text):
    ax.add_patch(FancyBboxPatch((x, y), w, 9, boxstyle="round,pad=0,rounding_size=1",
                 fc=tint(ACCENT, 0.90), ec=ACCENT, lw=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), 1.0, 9, boxstyle="square,pad=0",
                 fc=ACCENT, ec="none", zorder=3))
    ax.text(x + 3, y + 4.5, text, ha="left", va="center", fontsize=8.6,
            color=INK, zorder=4, wrap=True, linespacing=1.5)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25,
                facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


# =============================================================================
# 1. SYSTEM ARCHITECTURE
# =============================================================================
def fig_architecture():
    fig, ax = new_fig(18, 10)
    W = dict(top=97, right=176)
    header(ax, W, "System Architecture — ROS 2 Node Graph",
           "Sensors → perception → behavior-tree brain → navigation → thruster PWM. Wires labeled with principal ROS topics (simplified).")
    legend(ax, 4, 86, ["sensor", "perc", "mission", "nav", "control"])

    colx = [4, 40, 76, 110, 146]
    cw = 28
    # column headers
    heads = ["SENSING", "PERCEPTION", "MISSION BRAIN", "NAVIGATION", "CONTROL / ACTUATION"]
    for i, hd in enumerate(heads):
        ax.text(colx[i] + cw / 2, 82, hd, ha="center", va="bottom", fontsize=8.6,
                color=MUTED, family="monospace", fontweight="bold")

    # sensing
    s1 = card(ax, colx[0], 66, cw, 13, "ZED 2i stereo", "RGB · depth · VSLAM odom", "sensor")
    s2 = card(ax, colx[0], 50, cw, 13, "IMU / KVH FOG", "heading (truth)", "sensor")
    s3 = card(ax, colx[0], 34, cw, 13, "MAVROS ↔ ArduSub", "depth · state · RC", "sensor")
    # perception
    p1 = card(ax, colx[1], 66, cw, 13, "YOLO-seg detector", "8 classes + masks", "perc", key=True)
    p2 = card(ax, colx[1], 50, cw, 13, "Slalom pole detector", "red + white pipes", "perc", key=True)
    p3 = card(ax, colx[1], 34, cw, 13, "Octagon table det.", "sentinel cue", "perc")
    p4 = card(ax, colx[1], 18, cw, 13, "Target tracker", "IoU + depth fusion", "perc", key=True)
    # mission
    m1 = card(ax, colx[2], 58, cw, 14, "Mission node", "py_trees · blackboard", "mission", key=True)
    m2 = card(ax, colx[2], 40, cw, 13, "Nav tier manager", "VSLAM/VIO/DR", "mission")
    # navigation
    n1 = card(ax, colx[3], 66, cw, 12, "Vision servo", "crab-walk strafe", "nav")
    n2 = card(ax, colx[3], 52, cw, 12, "Waypoint nav", "surge/sway", "nav")
    n3 = card(ax, colx[3], 38, cw, 12, "Dead reckoning", "timed thrust + FOG", "nav")
    n4 = card(ax, colx[3], 24, cw, 12, "Yaw / depth PID", "closed loop", "nav")
    # control
    c1 = card(ax, colx[4], 60, cw, 12, "Mode manager", "arm · mode", "control")
    c2 = card(ax, colx[4], 46, cw, 12, "RC override", "→ PWM 1100/1500/1900", "control")
    c3 = card(ax, colx[4], 30, cw, 13, "Thrusters + servos", "8× + torpedo/marker", "control")

    arrow(ax, Rt(s1), L(p1), "zed/*")
    arrow(ax, Rt(s1), (p4["x"], p4["y"] + p4["h"] - 3), "depth", rad=-0.18)
    arrow(ax, Rt(s2), (m1["x"], m1["y"] + 3), "imu", rad=-0.12)
    arrow(ax, Rt(s3), L(p3), "state", rad=0.05)
    arrow(ax, B(p1), T(p4), "detections")
    arrow(ax, Rt(p2), (p4["x"], p4["y"] + p4["h"]), rad=-0.2)
    arrow(ax, Rt(p3), (p4["x"], p4["y"] + 3), rad=-0.15)
    arrow(ax, Rt(p4), (m1["x"], m1["y"] + 2), "tracked_targets", rad=-0.1)
    arrow(ax, Rt(m1), L(n1), "cmd_vel", rad=0.12)
    arrow(ax, Rt(m1), L(n4), "target_depth", rad=-0.12)
    for n in (n1, n2, n3, n4):
        arrow(ax, Rt(n), (c2["x"], c2["y"] + c2["h"] / 2 + (T(n)[1]-52)/6), rad=0.05)
    arrow(ax, Rt(m1), (c1["x"], c1["y"]+2), "arm/mode", rad=0.34)
    arrow(ax, B(c2), T(c3), "OverrideRCIn")

    note(ax, 4, 4, 168,
         "Heading is owned by the FOG (never the ZED), and safety-critical logic lives in the mission node — not inside the tree — so a single sensor dropout degrades gracefully instead of stalling the run.")
    save(fig, "01_system_architecture.png")


# =============================================================================
# 2. MISSION FLOW (behavior tree)
# =============================================================================
def fig_mission():
    fig, ax = new_fig(18, 10)
    W = dict(top=97, right=176)
    header(ax, W, "Mission Flow — py_trees Behavior Tree",
           "Root Sequence ticks at 10 Hz. Each task wrapped in FailureIsSuccess so a missed prop advances the run. Safety supervised outside the tree.")

    # root bar
    ax.add_patch(FancyBboxPatch((4, 78), 168, 6, boxstyle="round,pad=0,rounding_size=1",
                 fc=C["mission"], ec="none", zorder=2))
    ax.text(7, 81, "▣  MainMission", ha="left", va="center", color="white",
            fontsize=11, fontweight="bold", zorder=3)
    ax.text(169, 81, "Sequence · memory", ha="right", va="center", color="white",
            fontsize=8.5, family="monospace", zorder=3, alpha=0.9)

    stages = [
        ("Pre-dive", "Arm → AltHold →\nSubmerge → Stabilize", "mission", False),
        ("1 · Gate", "role align · record\ndivider · style spin", "mission", True),
        ("2 · Slalom", "strafe past 3 pipes\nheading locked", "mission", True),
        ("3 · Bins", "id by symbol\ndrop marker", "mission", True),
        ("4 · Torpedoes", "align hole (circle)\nfire", "mission", True),
        ("5 · Octagon", "advance blind\nsurface inside", "mission", True),
        ("6 · Return home", "dead-reckon to\ngate · cross back", "mission", True),
        ("Barrel roll", "MANUAL finale\n(kills FOG ref)", "control", False),
    ]
    n = len(stages)
    gap = 3.0
    cw = (168 - gap * (n - 1)) / n
    y = 56
    cards = []
    for i, (t, s, cat, key) in enumerate(stages):
        x = 4 + i * (cw + gap)
        c = card(ax, x, y, cw, 15, t, s, cat, key=key, tsize=9.2, ssize=7.0)
        cards.append(c)
    for i in range(n - 1):
        arrow(ax, Rt(cards[i]), L(cards[i + 1]))
    # resilient bracket note
    ax.annotate("", xy=(Rt(cards[6])[0], 72.5), xytext=(L(cards[1])[0], 72.5),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.3,
                                connectionstyle="arc3,rad=0"))
    ax.text((L(cards[1])[0]+Rt(cards[6])[0])/2, 74.5, "resilient  —  FailureIsSuccess (a missed task never aborts the run)",
            ha="center", va="bottom", fontsize=8, color=ACCENT, family="monospace")

    # safety supervisor strip
    ax.text(4, 44, "SAFETY SUPERVISOR", fontsize=9, color=ACCENT, family="monospace",
            fontweight="bold")
    ax.text(46, 44, "node-level · runs every tick · kept out of the tree", fontsize=8,
            color=MUTED)
    steps = [
        ("Watch", "mission timeout 900 s\n& tree status"),
        ("Surface", "until depth < 0.3 m\nor 30 s"),
        ("Disarm", "SetBool → false"),
        ("Shutdown", "rclpy.shutdown →\nlaunch teardown"),
    ]
    sw = (168 - 3 * 3) / 4
    sy = 26
    sc = []
    for i, (t, s) in enumerate(steps):
        x = 4 + i * (sw + 3)
        ax.add_patch(FancyBboxPatch((x + 0.5, sy - 0.6), sw, 13,
                     boxstyle="round,pad=0,rounding_size=1", fc=SHADOW, ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, sy), sw, 13, boxstyle="round,pad=0,rounding_size=1",
                     fc=tint(ACCENT, 0.90), ec=ACCENT, lw=1.4, zorder=2))
        ax.add_patch(Circle((x + 3.5, sy + 9.3), 1.8, fc=ACCENT, ec="none", zorder=3))
        ax.text(x + 3.5, sy + 9.3, str(i + 1), ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)
        ax.text(x + 6.5, sy + 9.3, t, ha="left", va="center", fontsize=10,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + 3, sy + 5.2, s, ha="left", va="top", fontsize=7.6, color=MUTED,
                family="monospace", zorder=4, linespacing=1.4)
        sc.append((x, sw))
    for i in range(3):
        x0 = sc[i][0] + sc[i][1]
        x1 = sc[i + 1][0]
        arrow(ax, (x0, sy + 6.5), (x1, sy + 6.5))

    note(ax, 4, 8, 168,
         "The mission commits irreversible decisions (which gate half, when to fire) on single detections — so timeout/emergency handling lives in the node, where a behavior-tree parallel can't fight the mission branch for the thrusters.")
    save(fig, "02_mission_flow.png")


# =============================================================================
# 3. CV — YOLO-SEG PIPELINE
# =============================================================================
def fig_cv():
    fig, ax = new_fig(18, 7)
    W = dict(top=64, right=176)
    header(ax, W, "Computer Vision — YOLO-seg on TensorRT",
           "Instance segmentation, not just boxes. Masks decoded from prototype coefficients; the mask centroid becomes the alignment point.")

    stages = [
        ("ZED rectified RGB", "rgb/color/rect/image", "sensor", False),
        ("Letterbox preprocess", "→ 640² · pad 114\nBGR→RGB · /255 · CHW", "perc", False),
        ("TensorRT inference", "TRT10 tensor-name API\nout0: dets (4+nc+nm)\nout1: mask protos", "perc", True),
        ("Decode + filter", "conf → per-class NMS\nmask = σ(coeffs·protos)\n→ area · contour · centroid", "perc", True),
        ("DetectionArray", "bbox · centroid\nmask_area · polygon", "perc", False),
    ]
    cw, gap, h = 28, 6.8, 22
    x = 4
    prev = None
    labels = ["", "blob", "2 out", "dets", ""]
    for i, (t, s, cat, key) in enumerate(stages):
        c = card(ax, x, 24, cw, h, t, s, cat, key=key, tsize=9.6, ssize=7.2)
        if prev:
            arrow(ax, Rt(prev), L(c), labels[i])
        prev = c
        x += cw + gap

    note(ax, 4, 4, 168,
         "Guardrail: at load the node verifies the engine's feature width equals 4 + num_classes + mask_coeffs and refuses to publish garbage from a mismatched (e.g. 80-class COCO) engine — the #1 cause of silent detection failure.")
    save(fig, "03_cv_yolo_pipeline.png")


# =============================================================================
# 4. PERCEPTION FUSION
# =============================================================================
def fig_fusion():
    fig, ax = new_fig(16, 9)
    W = dict(top=83, right=156)
    header(ax, W, "Perception Fusion — Three Detectors, One Stream",
           "The trained model is blind to unlabeled structures. We detect those by structure and color, merged under sentinel IDs above the model's range.")
    legend(ax, 4, 70, ["perc", "mission"])

    s1 = card(ax, 4, 53, 58, 15, "YOLO-seg model", "8 trained classes: blood · buoy · compass\ncircle · fire · hammer_and_wrench · slalom · sos",
              "perc", key=True, ssize=7.4)
    s2 = card(ax, 4, 35, 58, 15, "Slalom pole detector", "CLAHE→Canny→vert. morphology · shape gates\nHSV red/white tag · M-of-N temporal filter",
              "perc", key=True, ssize=7.4)
    s3 = card(ax, 4, 17, 58, 15, "Octagon table detector", "under-octagon capability-matrix board cue",
              "perc", ssize=7.6)

    merge = card(ax, 82, 37, 40, 16, "Target tracker", "unifies trained +\nsentinel detections\n+ depth & track IDs",
                 "perc", key=True, ssize=7.6)
    ax.text(82, 32, "sentinel classes:", fontsize=7.6, color=MUTED, family="monospace", va="center")
    sent1 = card(ax, 82, 24, 40, 6, "white_pole · id 8", "", "perc", tsize=8.6)
    sent2 = card(ax, 82, 16, 40, 6, "octagon_table · id 9", "", "perc", tsize=8.6)

    dest = card(ax, 132, 37, 22, 16, "Mission\nbehaviors", "key off\nclass_name", "mission", ssize=7.4)

    arrow(ax, Rt(s1), (merge["x"], merge["y"] + merge["h"] - 3), rad=-0.12)
    arrow(ax, Rt(s2), L(merge), rad=0.0)
    arrow(ax, Rt(s3), (merge["x"], merge["y"] + 3), rad=0.12)
    arrow(ax, Rt(merge), L(dest), "tracked_targets")

    note(ax, 4, 4, 150,
         "The payoff: the sub used to drive into white slalom pipes it literally could not see. Detecting a pole by BEING a pole (a tall high-contrast vertical bar) survives underwater red attenuation and earns the divider-side bonus for free.")
    save(fig, "04_perception_fusion.png")


# =============================================================================
# 5. TARGET TRACKER / DEPTH FUSION
# =============================================================================
def fig_tracker():
    fig, ax = new_fig(18, 7.5)
    W = dict(top=69, right=176)
    header(ax, W, "Target Tracker — Depth Fusion & Temporal Confirmation",
           "Raw detections are noisy and rangeless. The tracker associates across frames, smooths geometry, samples true range, and confirms before publishing.")

    i1 = card(ax, 4, 44, 26, 11, "Detections", "/hightide/detections", "perc", tsize=9.4)
    i2 = card(ax, 4, 30, 26, 11, "ZED depth map", "32FC1 depth_reg", "sensor", tsize=9.4)

    steps = [
        ("Purge stale", "age > 1.0 s\n→ drop", "perc", False),
        ("IoU matching", "best overlap ≥ 0.3\nsame class", "perc", True),
        ("EMA smoothing", "bbox & centroid\nα = 0.3", "perc", False),
        ("Sample range", "median of patch\nEMA 0.7/0.3", "perc", True),
        ("Hit gate", "hit_count ≥ 3\nbefore publish", "perc", True),
        ("tracked_targets", "2D + range\n+ track_id", "mission", False),
    ]
    x = 36
    cw, gap = 20, 4
    prev = None
    for i, (t, s, cat, key) in enumerate(steps):
        c = card(ax, x, 33, cw, 15, t, s, cat, key=key, tsize=8.6, ssize=7.0)
        if prev:
            arrow(ax, Rt(prev), L(c))
        else:
            arrow(ax, Rt(i1), (c["x"], c["y"] + c["h"] - 3), rad=-0.12)
            arrow(ax, Rt(i2), (c["x"], c["y"] + 3), rad=0.12)
        prev = c
        x += cw + gap

    note(ax, 4, 5, 168,
         "Why the hit gate matters: the mission fires torpedoes and picks bins on a single published detection. Requiring 3 confirming frames (~0.1–0.2 s) means one YOLO false positive can never trigger an actuator. Median-of-patch depth rejects the invalid/zero pixels the raw map is full of.")
    save(fig, "05_target_tracker.png")


# =============================================================================
# 6. NAVIGATION TIERS
# =============================================================================
def fig_tiers():
    fig, ax = new_fig(14, 8.6)
    W = dict(top=82, right=136)
    header(ax, W, "Navigation Tiers — Graceful SLAM Fallback",
           "Localization confidence derived from ZED odometry covariance, degraded in three tiers. Heading always comes from the FOG, not vision.")

    tiers = [
        ("TIER 1", "VSLAM", "confidence > 0.80",
         "Full visual SLAM with spatial memory — closed-loop waypoint nav. Best accuracy.", "perc", True),
        ("TIER 2", "VIO", "0.30 – 0.80",
         "Visual-inertial odometry — relative motion holds when spatial memory is weak.", "nav", False),
        ("TIER 3", "Dead reckoning", "conf < 0.30  or  stale",
         "Open-loop timed thrust with the FOG holding heading — navigate blind through a dropout.", "control", True),
    ]
    y = 58
    for tl, tn, cond, uses, cat, key in tiers:
        color = C[cat]
        ax.add_patch(FancyBboxPatch((4.6, y - 0.7), 128, 14,
                     boxstyle="round,pad=0,rounding_size=1", fc=SHADOW, ec="none", zorder=1))
        if key:
            ax.add_patch(FancyBboxPatch((3.4, y - 1.0), 130.4, 15.4,
                         boxstyle="round,pad=0,rounding_size=1.4", fc=tint(ACCENT, 0.8),
                         ec=ACCENT, lw=1.4, zorder=1.5))
        ax.add_patch(FancyBboxPatch((4, y), 128, 14, boxstyle="round,pad=0,rounding_size=1",
                     fc=PAGE, ec=BORDER, lw=1.2, zorder=2))
        # colored band
        ax.add_patch(FancyBboxPatch((4, y), 26, 14, boxstyle="round,pad=0,rounding_size=1",
                     fc=color, ec="none", zorder=3))
        ax.add_patch(plt.Rectangle((26, y), 3, 14, fc=color, ec="none", zorder=3))
        ax.text(17, y + 9.2, tl, ha="center", va="center", color="white", fontsize=8,
                family="monospace", zorder=4, alpha=0.92)
        ax.text(17, y + 4.6, tn, ha="center", va="center", color="white", fontsize=12.5,
                fontweight="bold", zorder=4)
        ax.add_patch(FancyBboxPatch((33, y + 4.2), 32, 5.6, boxstyle="round,pad=0,rounding_size=0.8",
                     fc=tint(color, 0.88), ec=color, lw=1.0, zorder=4))
        ax.text(49, y + 7.0, cond, ha="center", va="center", fontsize=8.6, color=INK,
                family="monospace", zorder=5)
        ax.text(69, y + 7.0, uses, ha="left", va="center", fontsize=8.8, color=MUTED,
                zorder=4, wrap=True)
        if key:
            ax.text(130, y + 12, "★", ha="right", va="top", fontsize=10, color=ACCENT,
                    fontweight="bold", zorder=5)
        y -= 17

    note(ax, 4, 4, 128,
         "Field-hardened default: covariance→confidence mapping and auto tier-switching are implemented but pinned to VSLAM so the manager never fights the stack mid-run; /hightide/set_navigation_tier forces any tier for in-water testing.")
    save(fig, "06_navigation_tiers.png")


if __name__ == "__main__":
    fig_architecture()
    fig_mission()
    fig_cv()
    fig_fusion()
    fig_tracker()
    fig_tiers()
    print("done ->", OUT)
