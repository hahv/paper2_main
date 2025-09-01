import time
import torch
import cv2
from PIL import Image
import timm
import numpy as np
from halib import *
from halib.filetype import csvfile
from collections import deque
from torchvision import transforms
from _archived.zz_config import FullConfig

# Constants
FD_CLASS_FIRE = 0
FD_CLASS_SMOKE = 1

class FireDetector:

    def __init__(self, cfg: FullConfig):
        self.cfg = cfg
        self.temporalStabilizationEnable = cfg.expConfig.enabled
        self.drawBlockDebug = cfg.expConfig.draw_block_debug
        # save masks when running temporal stabilization
        self.save_masks = (cfg.expConfig.save_masks and self.temporalStabilizationEnable)
        if self.temporalStabilizationEnable:
            self.temporalMethod = cfg.expConfig.methodUsed()
            self.windowSize = self.temporalMethod.params.get('window_size', 5)
            self.grid_size = self.temporalMethod.params.get('grid_size', 4)
            self.diff_frame_threshold = self.temporalMethod.params.get('diff_frame_threshold', 1)
            self.impack_plus_one = self.temporalMethod.params.get('impack_plus_one', 5)
            self.mask_threshold = self.temporalMethod.params.get('mask_threshold', 10)
            self.roi_threshold = self.temporalMethod.params.get('roi_threshold', 200)
            self.minimum_size = self.temporalMethod.params.get('minimum_size', 0.75)
            self.nonZeroTH = self.temporalMethod.params.get('nonZeroTH', 200)
        else:
            self.temporalMethod = None

        self.fireHistories = deque(maxlen=self.windowSize)
        self.smokeHistories = deque(maxlen=self.windowSize)

        if self.temporalStabilizationEnable:
            self.prev_frame = None
            self.deltaMasks = None
        # define NET_WIDTH_FD 640   /// net width for fd
        # define NET_HEIGHT_FD 360  /// net height for fd
        self.model = None
        NET_WIDTH_FD = 640
        NET_HEIGHT_FD = 360
        self.rowStep = int(NET_HEIGHT_FD / self.grid_size)
        self.colStep = int(NET_WIDTH_FD / self.grid_size)

    # ! should be overridden by the subclass if needed
    def custom_load_model(self):
        """Custom method to load the model if needed."""
        return timm.create_model(
                self.cfg.model.base_timm_model,
                pretrained=False,
                num_classes=len(self.cfg.model.class_names),
                checkpoint_path=self.cfg.model.model_path,
            )
    def loadModel(self):
        if self.model is None:
            self.model = self.custom_load_model()
            self.model.eval()
            # convert to suitable device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        return self.model

    def _infer(self, frame):
        assert self.model is not None, "Model is not loaded."
        model_name = fs.get_file_name(self.cfg.model.model_path, split_file_ext=True)[0]
        if "prof" in model_name.lower():
            transform = transforms.Compose(
                [
                    transforms.Resize((360, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        frame = transform(frame).unsqueeze(0)  # Add batch dimension
        # move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame = frame.to(device)
        with torch.no_grad():
            outputs = self.model(frame)

        labelIdx = torch.argmax(outputs, dim=1).item()
        # convert outputs to numpy array
        outputs = outputs.cpu().numpy().squeeze()  # Remove batch dimension
        outputs = outputs.tolist()  # Convert to list for easier handling

        classNames = self.cfg.model.class_names
        assert labelIdx < len(classNames), "Class index out of range."
        className = classNames[labelIdx]

        return outputs, labelIdx, className

    def setPreFrame(self, dst, src):
        dst[:] = cv2.resize(src, (224, 224))  # Example resiz

    def runModelSingle(self, roiFrame):
        fireFlag = 0
        smokeFlag = 0

        # Preprocess
        if roiFrame.shape[:2] == (224, 224):
            preFrame = roiFrame
        else:
            preFrame = cv2.resize(roiFrame, (224, 224))

        # Dummy inference
        classScores = np.random.rand(self.numClasses).astype(np.float32)

        classID = int(np.argmax(classScores))
        if classID == FD_CLASS_FIRE:
            fireFlag = 1
            smokeFlag = 1
        elif classID == FD_CLASS_SMOKE:
            smokeFlag = 1

        fireHist = self.fireHistories
        fireHist.append(fireFlag)
        fireProb = sum(fireHist) / self.windowSize

        smokeHist = self.smokeHistories
        smokeHist.append(smokeFlag)
        smokeProb = sum(smokeHist) / self.windowSize

        return True

    def getColor(self, classIdx):
        import seaborn as sns

        # main palette
        palette = sns.color_palette(palette="tab10")
        color = palette[classIdx % len(palette)]
        # Convert to 255 scale
        r, g, b = color
        color_255 = (int(r * 255), int(g * 255), int(b * 255))
        return color_255

    def _init_csv_writer(self, fname, columns):
        """Initialize CSV writer for inference results."""
        dfmk = csvfile.DFCreator()
        table_name = f"{fname}_infer.csv"
        dfmk.create_table(table_name, columns)
        return dfmk

    def _init_video_writer(self, video_path, fname):
        """Initialize video writer for output video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()

        out_video_path = os.path.join(self.cfg.general.outdir, f"{fname}_out.mp4")
        return cv2.VideoWriter(out_video_path, fourcc, fps, frame_size), out_video_path

    def _init_csv_writer(self, fname, columns):
        """Initialize CSV writer for inference results."""
        dfmk = csvfile.DFCreator()
        table_name = f"{fname}_infer.csv"
        dfmk.create_table(table_name, columns)
        return dfmk

    def _init_video_writer(self, video_path, fname):
        """Initialize video writer for output video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()

        out_video_path = os.path.join(self.cfg.general.outdir, f"{fname}_out.mp4")
        return cv2.VideoWriter(out_video_path, fourcc, fps, frame_size), out_video_path

    def _init_mask_writer(self, video_path, fname):
        """Initialize mask writer for output masks."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()

        out_mask_path = os.path.join(self.cfg.general.outdir, f"{fname}_masks.mp4")
        return cv2.VideoWriter(out_mask_path, fourcc, fps, frame_size), out_mask_path

    def _log_progress(self, frame_idx, total_frames, do_infer=True):
        """Log processing progress to console."""
        percentage = (frame_idx / total_frames) * 100
        prefix = "Infer" if do_infer else "[red]Skip[/red]"
        console.print(
            f"{prefix} frame {frame_idx}/{total_frames} ({percentage:.2f}%)...",
            end="\r",
            highlight=False
        )

    def _bgr_to_rgb(self, bgr):
        """Convert BGR color to RGB."""
        return (bgr[2], bgr[1], bgr[0])

    def _save_csv_results(self, dfmk, fname, rows):
        """Save inference results to CSV."""
        table_name = f"{fname}_infer.csv"
        dfmk.insert_rows(table_name, rows)
        dfmk.fill_table_from_row_pool(table_name)
        outfile = os.path.abspath(os.path.join(self.cfg.general.outdir, table_name))
        dfmk[table_name].to_csv(outfile, index=False, sep=";")
        pprint_local_path(outfile, tag=table_name)

    def runModel(self, frame):
        if self.temporalStabilizationEnable:
            # run temporal stabilization
            shouldDoInfer, roi = self.temporalStabilization(frame)
            if not shouldDoInfer:
                return True
            roiFrame = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            self.runModelSingle(roiFrame)
        else:
            # no temporal stabilization, run model on the whole frame
            self.runModelSingle(frame)

        return True

    # ! this function fn(current_frame, previous_frame), return ROI that is likely to contain motions (including fire/smoke)
    def temporalStabilization(self, cframe):
        rows_as_height, cols_as_width = cframe.shape[:2]
        # Default outputs
        shouldDoInfer = False
        topLeft = (0, 0)
        rightBottom = (cols_as_width, rows_as_height)
        vis_meta_dict = None
        maskFrame = None

        # First frame or shape mismatch
        if self.prev_frame is None or self.prev_frame.shape != cframe.shape:
            self.prev_frame = cframe.copy()
            # mask frame is just black
            maskFrame = np.zeros_like(cframe, dtype=np.uint8)
            return shouldDoInfer, (topLeft, rightBottom), vis_meta_dict, maskFrame

        # Initialization
        if self.deltaMasks is None or self.deltaMasks.shape != cframe.shape[:2]:
            self.deltaMasks = np.zeros((rows_as_height, cols_as_width), dtype=np.uint8)
            self.minWs = int(self.minimum_size * cols_as_width)
            self.minHs = int(self.minimum_size * rows_as_height)

        # --- Step 1: Delta Computation ---
        delta = cv2.absdiff(self.prev_frame, cframe)
        delta_gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
        _, delta_bin = cv2.threshold(delta_gray, self.diff_frame_threshold,
                                    self.impack_plus_one, cv2.THRESH_BINARY)

        MASK_CLIP_MAX = 25
        self.deltaMasks = np.minimum(self.deltaMasks + delta_bin, MASK_CLIP_MAX).astype(np.uint8)
        self.deltaMasks = cv2.subtract(self.deltaMasks, 1)

        # Optional: Visualization of masks
        if self.save_masks:
            normalized = (self.deltaMasks.astype(np.float32) * 255 / MASK_CLIP_MAX).astype(np.uint8)
            colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            maskFrame = cv2.addWeighted(cframe, 0.7, colormap, 0.3, 0)

        # --- Step 2: Mask Comparison ---
        curMask = cv2.compare(self.deltaMasks, self.mask_threshold, cv2.CMP_GE)
        self.prev_frame = cframe.copy()

        # --- Step 3: Block Analysis ---
        x0, y0, x1, y1 = cols_as_width, rows_as_height, 0, 0
        nonZeroTotal, nonZeroInsideBox = 0, 0
        block_infos = []

        for row in range(0, rows_as_height, self.rowStep):
            for col in range(0, cols_as_width, self.colStep):
                h = min(self.rowStep, rows_as_height - row)
                w = min(self.colStep, cols_as_width - col)
                block = curMask[row:row + h, col:col + w]
                nonZero = cv2.countNonZero(block)

                nonZeroTotal += nonZero
                if nonZero < self.nonZeroTH:
                    continue

                nonZeroInsideBox += nonZero
                x0 = min(x0, col)
                y0 = min(y0, row)
                x1 = max(x1, col + w)
                y1 = max(y1, row + h)

                if self.drawBlockDebug:
                    block_infos.append((col, row, w, h, nonZero))

        if self.drawBlockDebug:
            vis_meta_dict = {
                "nonZeroCountTotal": nonZeroTotal,
                "nonZeroCountInside": nonZeroInsideBox,
                "block_infos": block_infos,
            }

        # --- Step 4: ROI Calculation ---
        if x0 < x1 and y0 < y1:
            roi_width = x1 - x0
            roi_height = y1 - y0

            # what if the ROI is too small
            if roi_width < self.minWs:
                x0 = max(0, min(cols_as_width - self.minWs, x0 - (self.minWs - roi_width) // 2))
                roi_width = self.minWs
            if roi_height < self.minHs:
                y0 = max(0, min(rows_as_height - self.minHs, y0 - (self.minHs - roi_height) // 2))
                roi_height = self.minHs

            x1 = x0 + roi_width
            y1 = y0 + roi_height
            shouldDoInfer = True
            topLeft = (x0, y0)
            rightBottom = (x1, y1)
        else:
            shouldDoInfer = False

        return shouldDoInfer, (topLeft, rightBottom), vis_meta_dict, maskFrame

    def _annotate_frame(self, frame_idx, frame_rgb, probs, vis_meta_dict=None, topLeft_rightBottom_tupple=None, extract_info=None):
        """Annotate frame with class names and probabilities; debug temporal stabilization info if available."""
        class_probs = None
        if probs:
            class_probs = list(zip(self.cfg.model.class_names, probs))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        label_values = [("FrameIdx", frame_idx)] + class_probs if class_probs else [("FrameIdx", frame_idx)]

        if self.drawBlockDebug and vis_meta_dict is not None:
            nonZeroCountTotal = vis_meta_dict.get("nonZeroCountTotal", 0)
            nonZeroCountInside = vis_meta_dict.get("nonZeroCountInside", 0)
            label_values.append(("DEBUG", "--Block Analysis"))
            label_values.append(("NonZeroTotal", nonZeroCountTotal))
            label_values.append(("NonZeroInside", nonZeroCountInside))
        if extract_info is not None and isinstance(extract_info, tuple) and len(extract_info) == 2:
            # COLORMAP_JET
            label_values.append((extract_info[0], extract_info[1]))

        num_lines = len(label_values)
        box_height = 30 + num_lines * 30
        overlay = frame_bgr.copy()

        # Draw semi-transparent black rectangle
        cv2.rectangle(overlay, (0, 0), (300, box_height), (0, 0, 0), thickness=-1)
        frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
        # Add text annotations
        for i, (label, value) in enumerate(label_values):
            color = self._bgr_to_rgb(self.getColor(i))
            text = f"{label}: {value:.2f}" if isinstance(value, float) else f"{label}: {value}"
            cv2.putText(
                frame_bgr,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # draw the ROI if available
        if self.drawBlockDebug and topLeft_rightBottom_tupple is not None:
            topLeft, rightBottom = topLeft_rightBottom_tupple
            color_yellow = self._bgr_to_rgb((0, 255, 255))  # Yellow color for ROI
            cv2.rectangle(frame_bgr, topLeft, rightBottom, color_yellow, 2)

        return frame_bgr

    def _process_video_frames(self, video_path, dfmk, vwriter_and_path, mask_vwriter_and_path):
        """Process each frame of the video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")
        console.print(
            f"[red]temporalStabilizationEnable: {self.temporalStabilizationEnable}[/red]"
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        rows = []
        num_limit_frame = self.cfg.infer.limit
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if num_limit_frame > 0 and frame_idx > num_limit_frame:
                    console.print(f"Reached frame limit: {num_limit_frame}. Stopping inference.")
                    time.sleep(2) # pause a bit to let the user see the message
                    break
                topLeft = (0, 0)
                rightBottom = (frame.shape[1], frame.shape[0])
                shouldDoInfer = True
                vis_meta_dict = None
                maskFrame = None
                if self.temporalStabilizationEnable:
                    shouldDoInfer, (
                        topLeft,
                        rightBottom,
                    ), vis_meta_dict, maskFrame = self.temporalStabilization(frame)

                self._log_progress(frame_idx, total_frames, do_infer=shouldDoInfer)
                logits = None
                label_idx = None
                label = None
                probs = None
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if shouldDoInfer:
                    pil_img = Image.fromarray(frame_rgb)
                    # Crop the frame to the ROI if needed
                    if topLeft != (0, 0) or rightBottom != (frame.shape[1], frame.shape[0]):
                        x0, y0 = topLeft
                        x1, y1 = rightBottom
                        pil_img = pil_img.crop((x0, y0, x1, y1))
                    logits, label_idx, label = self._infer(pil_img)
                    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy().tolist()

                if dfmk:
                    rows.append([video_path, frame_idx, self.cfg.model.class_names, logits, probs, label_idx, label])

                if vwriter_and_path:
                    topLeft_rightBottom_tupple = (topLeft, rightBottom) if self.drawBlockDebug and shouldDoInfer else None
                    frame_bgr = self._annotate_frame(frame_idx, frame_rgb, probs, vis_meta_dict, topLeft_rightBottom_tupple)
                    vwriter_and_path[0].write(frame_bgr)
                if mask_vwriter_and_path and maskFrame is not None:
                    # If maskFrame is None, we skip writing the mask frame
                    topLeft_rightBottom_tupple = (topLeft, rightBottom) if self.drawBlockDebug and shouldDoInfer else None
                    extract_info = ("HeatMap", "Blue(low), Green(medium), Red(high)")
                    maskFrame = self._annotate_frame(
                        frame_idx,
                        maskFrame,
                        probs,
                        vis_meta_dict,
                        topLeft_rightBottom_tupple,
                        extract_info
                    )
                    mask_vwriter_and_path[0].write(maskFrame)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return {"rows": rows, "out_video_path": vwriter_and_path[1] if vwriter_and_path else None}

    def infer_video(self, video_path):
        """Main function to process video inference."""
        assert self.model is not None, "Model is not loaded."
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        columns = [
            "video_path",
            "frame_idx",
            "class_names",
            "logits",
            "probs",
            "pred_label_idx",
            "pred_label",
        ]
        source_in_ourdir = os.path.join(
            self.cfg.general.outdir, fs.get_file_name(video_path, split_file_ext=False)
        )

        # copy the video to our output directory
        if not os.path.exists(source_in_ourdir):
            fs.copy_file(video_path, source_in_ourdir)
            pprint_local_path(source_in_ourdir, tag=f"{fname}_video.mp4")

        with ConsoleLog(f"{fname} inference"):
            dfmk = (
                self._init_csv_writer(fname, columns)
                if self.cfg.infer.save_results
                else None
            )
            vwriter_and_path = (
                self._init_video_writer(video_path, fname)
                if self.cfg.infer.save_out_video
                else None
            )
            # mask (for temporal stabilization visualization)
            mask_vwriter_and_path = (
                self._init_mask_writer(video_path, fname)
                if self.save_masks
                else None
            )

            results = self._process_video_frames(video_path, dfmk, vwriter_and_path, mask_vwriter_and_path)
            if dfmk is not None:
                self._save_csv_results(dfmk, fname, results["rows"])
            if vwriter_and_path is not None:
                vwriter_and_path[0].release()
                pprint_local_path(results["out_video_path"], tag=f"{fname}_out.mp4")
            if mask_vwriter_and_path is not None:
                mask_vwriter_and_path[0].release()
                pprint_local_path(results["out_mask_path"], tag=f"{fname}_masks.mp4")
