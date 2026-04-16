# +-----------------------------------------------------------------------+
# |                    fig_qualitative_success.pdf                        |
# |                                                                       |
# |        (a) SKIP — Static BG    (b) INFER — Fire    (c) INFER — Smoke   |
# |                                                                       |
# |  Raw   +----------------+    +------------+    +------------+        |
# |        |                |    |            |    |            |        |
# |        |  Empty room    |    | 🔥 Flames  |    | 💨 Haze    |        |
# |        |  No change     |    | Flickering |    | Diffusing  |        |
# |        |                |    |            |    |            |        |
# |        | [GREEN border] |    |[BLUE border]    |[BLUE border]        |
# |        +----------------+    +------------+    +------------+        |
# |                                                                       |
# |  Mask  +----------------+    +------------+    +------------+        |
# |        |                |    |            |    |            |        |
# |        |   ░░░░░░░░░░   |    | ██████████ |    | ░░███░░░░  |        |
# |        |   (near dark)  |    | (fully lit)|    |(partially  |        |
# |        |   no activity  |    | high accum.|    | lit)       |        |
# |        +----------------+    +------------+    +------------+        |
# |         s_t = 0  ✓ SKIP       s_t = 1  ✓ RUN   s_t = 1  ✓ RUN      |
# +-----------------------------------------------------------------------+


# +-----------------------------------------------------------------------+
# |                    fig_qualitative_failure.pdf                        |
# |                                                                       |
# |         (d) WRONGLY SKIPPED — Slow Smoke                             |
# |                    (e) FORCED RUN — Persistent Motion                |
# |                                                                       |
# |  Raw   +---------------------+    +---------------------+           |
# |        |                     |    |                     |           |
# |        | 💨 Barely visible   |    | 🚶 Walking person   |           |
# |        |    smoke onset      |    |    no fire/smoke    |           |
# |        |    early stage      |    |    continuous move  |           |
# |        |                     |    |                     |           |
# |        |   [RED border]      |    |  [ORANGE border]    |           |
# |        +---------------------+    +---------------------+           |
# |                                                                       |
# |  Mask  +---------------------+    +---------------------+           |
# |        |                     |    |                     |           |
# |        |  ░░░░░░░░░░░░░░░   |    | ████████████████    |           |
# |        |  (near dark —       |    | (saturated —        |           |
# |        |  accumulator        |    |  K_max reached,     |           |
# |        |  not triggered)     |    |  never resets)      |           |
# |        +---------------------+    +---------------------+           |
# |         s_t = 0  ✗ MISSED          s_t = 1  ✗ NO SAVINGS           |
# +-----------------------------------------------------------------------+