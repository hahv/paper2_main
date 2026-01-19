import cv2
import numpy as np
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
from enum import Enum


class OsdFmt(str, Enum):
    """Common format strings."""

    INT = "{}"
    FLOAT = "{:.2f}"
    FLOAT3 = "{:.3f}"
    PERCENT = "{:.1%}"
    BOOL = "{}"
    HEX = "0x{:04X}"

    @classmethod
    def resolve(cls, fmt_input: Any) -> str:
        if isinstance(fmt_input, OsdFmt):
            return fmt_input.value
        if isinstance(fmt_input, str):
            try:
                return cls[fmt_input.upper()].value
            except KeyError:
                return fmt_input
        return OsdFmt.INT.value


class RenderUtils:
    """
    Flexible OSD renderer with automatic layout and optional background.
    """

    @staticmethod
    def get_rainbow_color(idx: int) -> Tuple[int, int, int]:
        try:
            palette = sns.color_palette("gist_rainbow")
            c = palette[idx % len(palette)]
            return (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        except Exception:
            return (0, 255, 0)

    @staticmethod
    def _auto_fmt(value: Any) -> str:
        return OsdFmt.FLOAT.value if isinstance(value, float) else OsdFmt.INT.value

    @staticmethod
    def _parse_setting(
        key: str, setting: Any, idx: int
    ) -> Tuple[str, Optional[str], Optional[Tuple[int, int, int]], float, int]:
        """Returns: (alias, fmt, color, scale, thickness)"""

        alias = key
        fmt_input = None
        color = None
        scale = 0.7
        thickness = 2

        if isinstance(setting, str):
            alias = setting

        elif isinstance(setting, (tuple, list)):
            if len(setting) > 0:
                alias = setting[0]

            for item in setting[1:]:
                if isinstance(item, (OsdFmt, str)):
                    fmt_input = item
                elif isinstance(item, (int, float)) and item < 5:
                    scale = float(item)
                elif isinstance(item, (tuple, list)) and len(item) == 3:
                    color = tuple(item)

        fmt = OsdFmt.resolve(fmt_input) if fmt_input else None
        return alias, fmt, color, scale, thickness

    @classmethod
    def calculate_osd_box(
        cls,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        padding: int = 10,
        line_spacing: int = 5,
    ) -> Tuple[int, int]:
        """
        Calculates the width and height of the OSD box without drawing.
        Returns: (box_width, box_height)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ---------------- PHASE 1: PREPARE TEXT ----------------
        lines = []

        # Helper to process text (same logic as draw_osd)
        def process_entry(key, val, idx, cfg=None):
            if cfg is not None:
                alias, fmt, _, scale, thick = cls._parse_setting(key, cfg, idx)
                fmt = fmt or cls._auto_fmt(val)
            else:
                alias = key
                fmt = cls._auto_fmt(val)
                scale = 0.7
                thick = 2

            try:
                val_str = fmt.format(val)
            except Exception:
                val_str = str(val)

            return {
                "text": f"{alias}: {val_str}",
                "scale": scale,
                "thickness": thick,
            }

        if config:
            for idx, (k, cfg) in enumerate(config.items()):
                if k in data:
                    lines.append(process_entry(k, data[k], idx, cfg))
        else:
            for idx, (k, v) in enumerate(data.items()):
                lines.append(process_entry(k, v, idx))

        if not lines:
            return (0, 0)

        # ---------------- PHASE 2: MEASURE ----------------
        max_w = 0
        total_h = 0

        for line in lines:
            (w, h), baseline = cv2.getTextSize(
                line["text"], font, line["scale"], line["thickness"]
            )
            # Row height = text height + baseline + spacing
            row_h = h + baseline + line_spacing

            max_w = max(max_w, w)
            total_h += row_h

        # Calculate final box dimensions
        box_w = max_w + 2 * padding
        # Subtract one line_spacing because the last line doesn't need bottom spacing inside the box
        box_h = total_h + 2 * padding - line_spacing

        return box_h, box_w

    @classmethod
    def draw_osd(
        cls,
        frame: np.ndarray,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        pos: Tuple[int, int] = (10, 30),
        bg_opacity: float = 0.5,
        padding: int = 10,
        line_spacing: int = 5,
    ) -> np.ndarray:
        """
        Draws multi-line OSD with auto-sized background box.
        If bg_opacity <= 0, background is not drawn.
        """

        vis = frame.copy()
        x_start, y_start = pos
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ---------------- PHASE 1: PREPARE TEXT ----------------
        lines = []

        def process_entry(key, val, idx, cfg=None):
            if cfg is not None:
                alias, fmt, color, scale, thick = cls._parse_setting(key, cfg, idx)
                fmt = fmt or cls._auto_fmt(val)
                color = color or cls.get_rainbow_color(idx)
            else:
                alias = key
                fmt = cls._auto_fmt(val)
                color = cls.get_rainbow_color(idx)
                scale = 0.7
                thick = 2

            try:
                val_str = fmt.format(val)
            except Exception:
                val_str = str(val)

            return {
                "text": f"{alias}: {val_str}",
                "color": color,
                "scale": scale,
                "thickness": thick,
            }

        if config:
            for idx, (k, cfg) in enumerate(config.items()):
                if k in data:
                    lines.append(process_entry(k, data[k], idx, cfg))
        else:
            for idx, (k, v) in enumerate(data.items()):
                lines.append(process_entry(k, v, idx))

        if not lines:
            return vis

        # ---------------- PHASE 2: MEASURE ----------------
        max_w = 0
        total_h = 0

        for line in lines:
            (w, h), baseline = cv2.getTextSize(
                line["text"], font, line["scale"], line["thickness"]
            )
            line["w"] = w
            line["h"] = h
            line["baseline"] = baseline
            line["row_h"] = h + baseline + line_spacing

            max_w = max(max_w, w)
            total_h += line["row_h"]

        box_w = max_w + 2 * padding
        box_h = total_h + 2 * padding - line_spacing

        # ---------------- PHASE 3: DRAW ----------------
        if bg_opacity > 0:
            overlay = vis.copy()
            cv2.rectangle(
                overlay,
                (x_start, y_start),
                (x_start + box_w, y_start + box_h),
                (0, 0, 0),
                -1,
            )
            vis = cv2.addWeighted(overlay, bg_opacity, vis, 1 - bg_opacity, 0)

        curr_y = y_start + padding

        for line in lines:
            draw_y = curr_y + line["h"]

            cv2.putText(
                vis,
                line["text"],
                (x_start + padding, draw_y),
                font,
                line["scale"],
                line["color"],
                line["thickness"],
                cv2.LINE_AA,
            )
            curr_y += line["row_h"]

        return vis


# ---------------- DEMO ----------------
def test():
    img = np.zeros((400, 500, 3), dtype=np.uint8) + 50

    results = {
        "frame_idx": 105,
        "fps": 24.532,
        "fire_prob": 0.98,
        "smoke_prob": 0.12,
        "status": "DANGER",
    }

    img_auto = RenderUtils.draw_osd(img.copy(), results)

    cfg = {
        "fps": ("FPS", "{:.1f}"),
        "status": ("System", (0, 0, 255)),
        "fire_prob": ("Fire", OsdFmt.PERCENT, (0, 0, 255)),
        "smoke_prob": ("Smoke", OsdFmt.PERCENT),
    }

    img_cfg = RenderUtils.draw_osd(img.copy(), results, cfg, bg_opacity=0.5)
    img_no_bg = RenderUtils.draw_osd(img.copy(), results, cfg, bg_opacity=0)

    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    os.makedirs("_test_out", exist_ok=True)
    os.chdir("_test_out")
    cv2.imwrite("test_auto.jpg", img_auto)
    cv2.imwrite("test_cfg.jpg", img_cfg)
    cv2.imwrite("test_no_bg.jpg", img_no_bg)

    print("Saved test images.")


if __name__ == "__main__":
    test()
