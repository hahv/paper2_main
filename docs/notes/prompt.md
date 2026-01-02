Context: fire/smoke detection for video from static cameras

Normal Pipeline:
1. Read video frames from a static camera (or video file).
2. preprocess frames and feed into the DL model (classification: fire/smoke OR None label).
3. alarm if fire/smoke detected.

One drawback of this pipeline is that it always processes every frame, which can be inefficient. What if we could skip frames that are unlikely to contain fire or smoke, e.g., when the camera is looking at a static scene without any movement?

Optimized Pipeline:
1. Read video frames from a static camera (or video file).
2. Pre-check to detect the scene that is likely no fire/smoke present.
3. If no-fire/smoke scene detected, skip processing the frame.
4. If scene is likely to contain fire/smoke, preprocess the frame and feed into the DL model (classification: fire/smoke OR None label).
5. Alarm if fire/smoke detected.

Suggest as many as the pre-check method to detect the scene that is likely no fire/smoke present (use only video data, no additional sensors)
