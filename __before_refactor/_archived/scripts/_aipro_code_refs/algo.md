START
  |
  v
[Get reference frame (tFrame) for video channel (vchID)]
  |
  v
[Is tFrame empty or size mismatch with current frame?]
  | Yes
  |----> [Clone current frame to tFrame]
  |       |
  |       v
  |      [Set ROI to full frame (0, 0, frame.cols, frame.rows)]
  |       |
  |       v
  |      [Return false (skip processing)]
  |
  v No
[Get minW, minH, and deltaMask for vchID]
  |
  v
[Is deltaMask empty or size mismatch with frame?]
  | Yes
  |----> [Initialize deltaMask as zero matrix (frame size, CV_8UC1)]
  |       |
  |       v
  |      [Set minW, minH based on frame size]
  |       | If frame is 1920x1080:
  |       |   minW = 1280, minH = 720
  |       | Else:
  |       |   minW = 0.75 * frame.cols, minH = 0.75 * frame.rows
  |
  v No
[Compute absolute difference: absdiff(tFrame, frame, delta)]
  |
  v
[Convert delta to grayscale (COLOR_BGR2GRAY)]
  |
  v
[Apply binary threshold to delta: > DIFF_FRAME_TH (1) -> IMPACK_PLUS_ONE (5), else 0]
  |
  v
[Update deltaMask: deltaMask = deltaMask + delta]
  |
  v
[Cap deltaMask values at 25]
  |
  v
[Subtract 1 from deltaMask (decay)]
  |
  v
[Create binary curMask: deltaMask >= MASK_TH (10) -> 255, else 0]
  |
  v
[Copy current frame to tFrame (update reference)]
  |
  v
[Initialize variables: min_x0 = cols, min_y0 = rows, max_x1 = 0, max_y1 = 0]
[Initialize counters: nonZeroCountTotal = 0, nonZeroCountInside = 0]
  |
  v
[Loop over frame in blocks (rowStep, colStep)]
  | For each block (row, col):
  |   |
  |   v
  |  [Define block: Rect(col, row, blockWidth, blockHeight)]
  |   |
  |   v
  |  [Extract blockMask from curMask]
  |   |
  |   v
  |  [Count non-zero pixels in blockMask (nonZeroCount)]
  |   |
  |   v
  |  [Add nonZeroCount to nonZeroCountTotal]
  |   |
  |   v
  |  [Is nonZeroCount < ROI_TH (200)?]
  |   | Yes
  |   |----> [Skip to next block]
  |   |
  |   v No
  |  [Add nonZeroCount to nonZeroCountInside]
  |   |
  |   v
  |  [Update bounding box: min_x0, min_y0, max_x1, max_y1]
  |   |
  |   v
  |  [If drawBlockDebug is true]
  |   |----> [Draw red rectangle around block]
  |   |----> [Write nonZeroCount on frame at (col+10, row+20)]
  |
  v
[If drawBlockDebug is true]
 ம

System: * Today's date and time is 01:34 PM KST on Monday, July 07, 2025.