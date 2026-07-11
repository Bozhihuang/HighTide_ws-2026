# hightide_perception package

# Competition YOLO class mapping.
#
# These IDs/names MUST match the front-facing-camera (ffc) model's data.yaml
# exactly — the detector reads names from this dict, so the order/index is the
# contract with the trained weights. This is the 8-class ffc model (symbols +
# slalom red poles + octagon buoy + torpedo holes), NOT a structural-box model:
#   - no dedicated 'gate'/'bin'/'torpedo_board'/'octagon' boxes — those are
#     inferred from the role symbols / buoy that sit on them.
#   - 'slalom' == the RED slalom poles only (per the labeling guide), which is
#     exactly the divider we align to, so the red-side bonus still works.
#   - 'circle' == each torpedo hole (large vs small distinguished by box size).
CLASS_NAMES = {
    0: 'blood',              # blood-drop emoji — Search & Rescue role symbol
    1: 'buoy',               # octagon buoy border — octagon approach cue
    2: 'compass',            # compass emoji — Survey & Repair role symbol
    3: 'circle',             # a torpedo hole (incl. its red ring)
    4: 'fire',               # fire emoji — Survey & Repair role symbol
    5: 'hammer_and_wrench',  # hammer+wrench emoji — Survey & Repair role symbol
    6: 'slalom',             # RED slalom poles
    7: 'sos',                # SOS emoji — Search & Rescue role symbol
}
