import cv2
import numpy as np
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, Union
from enum import Enum
import os


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
    Uses dictionary-based configuration for styling.
    """

    # Default style values if not provided in the config dict
    DEFAULT_STYLE = {
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "label": None,  # Defaults to the data key
        "fmt": None,  # Defaults to auto-detection
        "color": None,  # Defaults to rainbow generation
        "scale": 0.7,
        "thickness": 2,
    }

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

    @classmethod
    def _parse_setting(
        cls, key: str, setting: Union[Dict[str, Any], str], idx: int
    ) -> Dict[str, Any]:
        """
        Parses a single configuration item.
        """
        # Start with defaults
        parsed = cls.DEFAULT_STYLE.copy()

        # 1. Handle Shorthand (String only)
        if isinstance(setting, str):
            parsed["label"] = setting

        # 2. Handle Dictionary Config
        elif isinstance(setting, dict):
            # Update defaults with provided values
            parsed.update(setting)

        # 3. Resolve final values
        if parsed["label"] is None:
            parsed["label"] = key

        if parsed["color"] is None:
            parsed["color"] = cls.get_rainbow_color(idx)

        if parsed["fmt"]:
            parsed["fmt"] = OsdFmt.resolve(parsed["fmt"])

        return parsed

    @classmethod
    def calculate_osd_box(
        cls,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        padding: int = 10,
        line_spacing: int = 5,
    ) -> Tuple[int, int]:
        """
        Calculates the (height, width) of the OSD box without drawing.
        """
        lines = cls._prepare_lines(data, config)

        if not lines:
            return (0, 0)

        # Measure
        max_w = 0
        total_h = 0

        for line in lines:
            # FIX: Use the line's specific font, not a global one
            (w, h), baseline = cv2.getTextSize(
                line["text"], line["font"], line["scale"], line["thickness"]
            )
            row_h = h + baseline + line_spacing
            max_w = max(max_w, w)
            total_h += row_h

        box_w = max_w + 2 * padding
        box_h = total_h + 2 * padding - line_spacing

        return box_h, box_w

    @classmethod
    def _prepare_lines(
        cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> list:
        """Helper to generate the list of text lines with their styles."""
        lines = []

        def create_line_data(key, val, idx, cfg=None):
            # Parse the style (merging defaults + config)
            style = cls._parse_setting(key, cfg or {}, idx)

            # Determine format
            fmt = style["fmt"] or cls._auto_fmt(val)

            try:
                val_str = fmt.format(val)
            except Exception:
                val_str = str(val)

            return {
                "text": f"{style['label']}: {val_str}",
                # FIX: Use the parsed font style, NOT hardcoded cv2.FONT_HERSHEY_SIMPLEX
                "font": style["font"],
                "color": style["color"],
                "scale": style["scale"],
                "thickness": style["thickness"],
            }

        if config:
            for idx, (k, cfg_item) in enumerate(config.items()):
                if k in data:
                    lines.append(create_line_data(k, data[k], idx, cfg_item))
        else:
            for idx, (k, v) in enumerate(data.items()):
                lines.append(create_line_data(k, v, idx))

        return lines

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
        Draws multi-line OSD.
        config: Dict[key_name, Dict] -> {"label": "Alias", "color": (255,0,0), ...}
        """
        vis = frame.copy()
        x_start, y_start = pos

        # 1. Prepare Text Lines
        lines = cls._prepare_lines(data, config)

        if not lines:
            return vis

        # 2. Measure for Background
        max_w = 0
        total_h = 0

        # We need to store measurements to avoid calling getTextSize twice
        measured_lines = []

        for line in lines:
            # FIX: Use line["font"] here so measurement matches drawing
            (w, h), baseline = cv2.getTextSize(
                line["text"], line["font"], line["scale"], line["thickness"]
            )
            line_h = h + baseline + line_spacing

            measured_lines.append(
                {
                    **line,
                    "w": w,
                    "h": h,  # text height only
                    "row_h": line_h,
                }
            )

            max_w = max(max_w, w)
            total_h += line_h

        box_w = max_w + 2 * padding
        box_h = total_h + 2 * padding - line_spacing

        # 3. Draw Background
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

        # 4. Draw Text
        curr_y = y_start + padding

        for line in measured_lines:
            # Text origin in OpenCV is bottom-left of the string
            draw_y = curr_y + line["h"]

            cv2.putText(
                vis,
                line["text"],
                (x_start + padding, draw_y),
                line["font"],  # Correctly using line-specific font
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

    data = {
        "frame_idx": 105,
        "fps": 24.532,
        "fire_prob": 0.98,
        "smoke_prob": 0.12,
        "status": "DANGER",
    }

    print("1. Testing Auto Mode (No Config)...")
    img_auto = RenderUtils.draw_osd(img.copy(), data)

    # New Dictionary-based Configuration
    data_render_cfg = {
        "fps": {
            "label": "FPS",
            "fmt": "{:.1f}",
            "scale": 0.4,
            "thickness": 1,
            "font": cv2.FONT_HERSHEY_COMPLEX_SMALL,  # <--- Custom font test
        },
        "status": {"label": "System", "color": (0, 0, 255), "thickness": 2},
        "fire_prob": {"label": "Fire", "fmt": OsdFmt.PERCENT, "color": (0, 0, 255)},
        "smoke_prob": {
            "label": "Smoke",
            "fmt": OsdFmt.PERCENT,
        },
    }

    print("2. Testing Custom Config Mode...")
    img_cfg = RenderUtils.draw_osd(img.copy(), data, data_render_cfg, bg_opacity=0.5)

    print("3. Testing No Background Mode...")
    img_no_bg = RenderUtils.draw_osd(img.copy(), data, data_render_cfg, bg_opacity=0)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    os.makedirs("_test_out", exist_ok=True)
    os.chdir("_test_out")

    cv2.imwrite("test_auto.jpg", img_auto)
    cv2.imwrite("test_cfg.jpg", img_cfg)
    cv2.imwrite("test_no_bg.jpg", img_no_bg)

    print(f"Saved test images to {os.getcwd()}")


if __name__ == "__main__":
    test()
