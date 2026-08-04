It is a very sharp observation. You have spotted a **hidden coupling** in your parameters.

While your current setup works, it is **fragile**. If you decide to optimize performance by lowering `scale_factor` (e.g., from `0.5` to `0.25`), your blocks suddenly cover **4x the area** in the real world. You essentially change your "detection sensitivity" just by trying to make the code faster.

Here is the breakdown of why this happens and two better alternatives to decouple them.

### 1. The Problem: "Effective" Block Size

Currently, `block_size` is defined in the **scaled (small)** domain.

- **Scenario A:** 1080p Video, `scale_factor=0.5` (540p), `block_size=32`.
- Real-world coverage: **64x64 pixels**.

- **Scenario B:** 1080p Video, `scale_factor=0.1` (108p), `block_size=32`.
- Real-world coverage: **320x320 pixels**.

In Scenario B, your grid is massive. A small fire or person might fit entirely inside one block, and if they don't trigger the threshold, you miss them. You shouldn't have to retune `block_size` every time you change `scale_factor`.

---

### 2. Better Alternative A: Define `grid_shape` (Rows, Cols)

Instead of defining the size of a block in pixels, define **how many blocks** you want across the screen. This guarantees your "granularity" stays the same regardless of resolution or scaling.

**New Params:**

- Remove: `block_size`
- Add: `grid_count` (e.g., `20` meaning a 20x20 grid, or just horizontal count).

**How to implement:**
You calculate `block_size` dynamically during initialization or the first frame.

```python
class BlockSkipProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.scale_factor = self.params.get("scale_factor", 0.5)
        # Instead of fixed pixels, we say "I want 32 blocks across the width"
        self.grid_width_count = self.params.get("grid_width_count", 32)
        self.block_size = None # Calculated later

    def _update_block_size(self, frame_w, frame_h):
        # Calculate scaled dimensions
        scaled_w = int(frame_w * self.scale_factor)

        # derive block_size dynamically
        self.block_size = int(scaled_w / self.grid_width_count)

        # Enforce a minimum safety size (e.g., don't go below 4px)
        self.block_size = max(4, self.block_size)

```

**Pros:** "Set and forget." Works on 720p and 4K videos identically.
**Cons:** Blocks might not be perfectly square if you lock aspect ratio, but usually negligible.

---

### 3. Better Alternative B: Define `block_size` in ORIGINAL Pixels

This is the most intuitive for humans. You look at your 1080p video and say, "I want to detect motion in 50x50 regions." The code handles the scaling math for you.

**New Params:**

- `scale_factor`: `0.5` (Speed tuning)
- `target_block_size`: `64` (Logic tuning - in original pixels)

**How to implement:**

```python
class BlockSkipProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.scale_factor = self.params.get("scale_factor", 0.5)

        # User implies: "64px on the original 1080p image"
        target_original_size = self.params.get("target_block_size", 64)

        # We calculate what that looks like on the small image
        # e.g. 64 * 0.5 = 32px actual block size
        self.block_size = int(target_original_size * self.scale_factor)

        # Safety clamp
        self.block_size = max(4, self.block_size)

```

**Pros:** Decouples Logic from Speed completely.
**Cons:** None, really. This is the recommended approach if you want fixed physical size detection.

### Recommendation

I recommend **Alternative B (Original Pixel Size)**.

It aligns best with how we think about computer vision problems: "I need to find objects roughly size X." You set that size once. Then, you can lower `scale_factor` as much as you want to speed up the system, and your code automatically shrinks the processing block size to maintain that same detection granularity.
