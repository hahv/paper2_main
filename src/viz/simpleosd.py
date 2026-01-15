import cv2
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, Union
from enum import Enum


# --- 1. ENUM DEFINITION ---
class OsdFmt(str, Enum):
    """
    Common format strings. Inheriting from str makes them behave like strings.
    """

    INT = "{}"
    FLOAT = "{:.2f}"
    FLOAT3 = "{:.3f}"
    PERCENT = "{:.1%}"
    TIME = "{:.4f}s"
    BOOL = "{}"
    HEX = "0x{:04X}"

    @classmethod
    def resolve(cls, fmt_input: Any) -> str:
        """Helper to resolve Enums, Strings, or Defaults."""
        if isinstance(fmt_input, OsdFmt):
            return fmt_input.value
        if isinstance(fmt_input, str):
            # Try to find a matching Enum name (case-insensitive), e.g., "float" -> OsdFmt.FLOAT
            try:
                return cls[fmt_input.upper()].value
            except KeyError:
                return fmt_input  # Return custom string like "{:.5f}"
        return OsdFmt.INT.value  # Default


# --- 2. OSD ENGINE ---
class SimpleOSD:
    def __init__(
        self,
        config: Dict[str, Any],
        default_pos=(10, 30),
        default_color=(0, 255, 0),
        default_scale=0.6,
        thickness=1,
        padding=5,
    ):
        self.config = config
        self.default_pos = default_pos
        self.default_color = default_color
        self.default_scale = default_scale
        self.thickness = thickness
        self.padding = padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_auto_fmt(self, value: Any) -> str:
        """Smart defaults if no format is provided."""
        if isinstance(value, float):
            return OsdFmt.FLOAT.value
        return OsdFmt.INT.value

    def draw(
        self,
        frame: np.ndarray,
        data_dict: Dict[str, Any],
        pos: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        x, y = pos if pos is not None else self.default_pos

        for key, settings in self.config.items():
            if key not in data_dict:
                continue

            val = data_dict[key]

            # --- 1. Smart Parsing ---
            # Defaults
            alias = key
            fmt_input = None
            scale = self.default_scale
            color = self.default_color

            # CASE A: Simple String -> {"accuracy": "Acc"}
            if isinstance(settings, str):
                alias = settings

            # CASE B: Configuration Tuple -> {"accuracy": ("Acc", ...)}
            elif isinstance(settings, (tuple, list)):
                if len(settings) > 0:
                    alias = settings[0]

                # Scan remaining items by TYPE (Order doesn't matter!)
                for item in settings[1:]:
                    if isinstance(item, (str, OsdFmt)):
                        fmt_input = item
                    elif isinstance(item, (int, float)):
                        scale = float(item)
                    elif isinstance(item, (tuple, list)):
                        color = item

            # --- 2. Resolve Format ---
            # Try to resolve input (Enum/Str), otherwise fallback to Auto-Detect
            fmt = OsdFmt.resolve(fmt_input)
            if fmt is None:
                fmt = self._get_auto_fmt(val)

            # --- 3. Render ---
            try:
                val_str = fmt.format(val)
            except:
                val_str = str(val)

            text = f"{alias}: {val_str}"

            # Compact Layout Calculation
            (w, h), baseline = cv2.getTextSize(text, self.font, scale, self.thickness)
            cv2.putText(
                frame,
                text,
                (x, y),
                self.font,
                scale,
                color,
                self.thickness,
                cv2.LINE_AA,
            )
            y += h + baseline + self.padding

        return frame


# --- TEST CASE ---
def test_clean_config():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # !!! DEMONSTRATION OF SIMPLE VS ADVANCED CONFIG !!!
    osd_config = {
        # 1. THE SIMPLEST CASE (What you asked for)
        # Just maps Key -> Alias.
        # Logic: Auto-detects float -> uses "{:.2f}" -> Green Color
        "model_score": "Score",
        # 2. Simple but with Enum for specific formatting
        "accuracy": ("Acc", OsdFmt.PERCENT),
        # 3. Complex: Alias + Enum + Color + Scale
        "alert_status": ("WARN", OsdFmt.BOOL, (0, 0, 255), 1.0),
    }

    osd = SimpleOSD(osd_config)

    data = {
        "model_score": 1234.5678,  # Will show: "Score: 1234.57"
        "accuracy": 0.985,  # Will show: "Acc: 98.5%"
        "alert_status": True,  # Will show: "WARN: True" (in Red, Big)
    }

    osd.draw(frame, data, pos=(20, 50))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "osd_output.png")
    cv2.imwrite(output_path, frame)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    test_clean_config()
