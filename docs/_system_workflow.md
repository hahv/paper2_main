+----------------+
|  Input:       |
| FrameBefore (FB)|
| FrameAfter(FA) |
| Video Frame Size (S)|
+----------------+
        |
        v
+----------------+
|  Divide FA and FB into Grid |
|  (9x9)              |
+----------------+
        |
        v
+----------------+
|  Detect Cell Changes  |
|  (FrameDiff or other) |
+----------------+
        |
        v
+----------------+
|  Calculate CellDIFF   |
|  (Single value call)  |
+----------------+
        |
        v
+----------------+
|  Mark Active Cells    |
|  (CellDIFF > Predefined_Threshold)|
+----------------+
        |
        v
+----------------+
|  Find Bounding Box   |
|  (Wrap active cells) 
if not enough active cells, use bbox that covers the active cells but expands to a predefined threshold (let say 80% of S)|
+----------------+
        |
        v
+----------------+
|  Crop FrameAfter    |
|  (Using bbox)        |
+----------------+
        |
        v
+----------------+
|  Feed to Detection Model|
+----------------+