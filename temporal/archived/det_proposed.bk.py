import cv2
import torch
from halib import *
from PIL import Image
from torchvision import transforms
from fire_temporal.det_base import DetectorBase
from fire_temporal.utils import *
from fire_temporal.config import Config
from halib.filetype import csvfile

from halib.research.perfcalc import (
    PerfCalc,
    validate_torch_metrics,
    valid_custom_fields,
)


class ProposedDetector(DetectorBase, PerfCalc):
    POSITIVE_LABEL = "positive"
    NEGATIVE_LABEL = "negative"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.methodUsed = cfg.method.methodUsed
        self.method_name = cfg.method.name
        if self.method_name == "temp_stabilize":
            params = self.methodUsed.params
            self.grid_size = params.get("grid_size", 4)
            self.diff_frame_thres = params.get("diff_frame_thres", 1)
            self.impack_plus_one = params.get("impack_plus_one", 5)
            self.mask_thres = params.get("mask_thres", 10)
            self.min_size = params.get("min_size", 0.75)
            self.non_zero_thres = params.get("non_zero_thres", 200)
            NET_WIDTH_FD = 640
            NET_HEIGHT_FD = 360
            self.rowStep = int(NET_HEIGHT_FD / self.grid_size)
            self.colStep = int(NET_WIDTH_FD / self.grid_size)

            # for debugging purposes
            self.drawBlockDebug = self.methodUsed.params["debug"]["draw_block_debug"]
            self.save_masks = self.methodUsed.params["debug"]["save_masks"]

            # for temporal stabilization (foreground motion accumulation)
            self.deltaMasks = None
            self.prev_frame = None

        elif self.method_name == "no_temp":
            pass  # current no additional parameters for no temporal stabilization
        elif self.method_name == "temp_tpt":
            params = self.methodUsed.params
            self.window_size = params.get("window_size", 20)
            self.persistence_thres = params.get("persistence_thres", 0.5)
        else:
            raise ValueError(
                f"Unknown method name: {self.method_name}. Supported methods: {self.cfg.expConfig.list_methods.keys()}"
            )

    def _init_mask_writer(self, video_path):
        """Initialize mask writer for output masks."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        self.out_mask_path = os.path.join(self.outdir, f"{fname}_out_masks.mp4")
        self.mask_writer = cv2.VideoWriter(self.out_mask_path, fourcc, fps, frame_size)

    # ! this function fn(current_frame, previous_frame), return ROI that is likely to contain motions (including fire/smoke)
    def _tempStabilize(self, cframe):
        rows_as_height, cols_as_width = cframe.shape[:2]
        # Default outputs
        shouldDoInfer = False
        topLeft = (0, 0)
        rightBottom = (cols_as_width, rows_as_height)
        vis_data_results = None
        maskFrame = None

        # First frame or shape mismatch
        if self.prev_frame is None or self.prev_frame.shape != cframe.shape:
            self.prev_frame = cframe.copy()
            # mask frame is just black
            maskFrame = np.zeros_like(cframe, dtype=np.uint8)
            return shouldDoInfer, (topLeft, rightBottom), vis_data_results, maskFrame

        # Initialization
        if self.deltaMasks is None or self.deltaMasks.shape != cframe.shape[:2]:
            self.deltaMasks = np.zeros((rows_as_height, cols_as_width), dtype=np.uint8)
            self.minWs = int(self.min_size * cols_as_width)
            self.minHs = int(self.min_size * rows_as_height)

        # --- Step 1: Delta Computation ---
        delta = cv2.absdiff(self.prev_frame, cframe)
        delta_gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
        _, delta_bin = cv2.threshold(
            delta_gray,
            self.diff_frame_thres,
            self.impack_plus_one,
            cv2.THRESH_BINARY,
        )

        MASK_CLIP_MAX = 25
        self.deltaMasks = np.minimum(self.deltaMasks + delta_bin, MASK_CLIP_MAX).astype(
            np.uint8
        )
        self.deltaMasks = cv2.subtract(self.deltaMasks, 1)

        # Optional: Visualization of masks
        if hasattr(self, "save_masks") and self.save_masks:
            normalized = (
                self.deltaMasks.astype(np.float32) * 255 / MASK_CLIP_MAX
            ).astype(np.uint8)
            colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            maskFrame = cv2.addWeighted(cframe, 0.7, colormap, 0.3, 0)

        # --- Step 2: Mask Comparison ---
        curMask = cv2.compare(self.deltaMasks, self.mask_thres, cv2.CMP_GE)
        self.prev_frame = cframe.copy()

        # --- Step 3: Block Analysis ---
        x0, y0, x1, y1 = cols_as_width, rows_as_height, 0, 0
        nonZeroTotal, nonZeroInsideBox = 0, 0
        block_infos = []

        for row in range(0, rows_as_height, self.rowStep):
            for col in range(0, cols_as_width, self.colStep):
                h = min(self.rowStep, rows_as_height - row)
                w = min(self.colStep, cols_as_width - col)
                block = curMask[row : row + h, col : col + w]
                nonZero = cv2.countNonZero(block)

                nonZeroTotal += nonZero
                if nonZero < self.non_zero_thres:
                    continue

                nonZeroInsideBox += nonZero
                x0 = min(x0, col)
                y0 = min(y0, row)
                x1 = max(x1, col + w)
                y1 = max(y1, row + h)
                # Update the mask with the current block
                if hasattr(self, "drawBlockDebug") and self.drawBlockDebug:
                    # pprint(f"Block at ({col}, {row}) with size ({w}, {h}) has non-zero count: {nonZero}")
                    block_infos.append((col, row, w, h, nonZero))

        if hasattr(self, "drawBlockDebug") and self.drawBlockDebug:
            # pprint(f"Total non-zero count: {nonZeroTotal}, Inside box: {nonZeroInsideBox}")
            vis_data_results = {
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
                x0 = max(
                    0,
                    min(cols_as_width - self.minWs, x0 - (self.minWs - roi_width) // 2),
                )
                roi_width = self.minWs
            if roi_height < self.minHs:
                y0 = max(
                    0,
                    min(
                        rows_as_height - self.minHs, y0 - (self.minHs - roi_height) // 2
                    ),
                )
                roi_height = self.minHs

            x1 = x0 + roi_width
            y1 = y0 + roi_height
            shouldDoInfer = True
            topLeft = (x0, y0)
            rightBottom = (x1, y1)
        else:
            shouldDoInfer = False

        return shouldDoInfer, (topLeft, rightBottom), vis_data_results, maskFrame

    def _draw_block_debug(self, frame_bgr, block_infos):
        """Draw debug blocks on the frame."""
        for col, row, w, h, nonZero in block_infos:
            color = (0, 0, 255)  # Red color for blocks
            cv2.rectangle(frame_bgr, (col, row), (col + w, row + h), color, 2)
            # Text to display
            text = f"{nonZero}"

            # Calculate rectangle center
            center_x = col + w // 2
            center_y = row + h // 2

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Compute bottom-left corner so text is centered
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2

            # Draw text
            cv2.putText(
                frame_bgr,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _annotate_frame(
        self,
        frame_bgr,  # ! make sure that this frame is in BGR format
        label_value_dict,  # {"label": value} pairs, e.g., {"FrameIdx": "1/100", etc.}
        vis_data_results=None,
    ):
        """Annotate frame with class names and probabilities; debug temporal stabilization info if available."""

        num_lines = len(label_value_dict)
        box_height = 30 + num_lines * 30
        overlay = frame_bgr.copy()

        # Draw semi-transparent black rectangle
        cv2.rectangle(overlay, (0, 0), (300, box_height), (0, 0, 0), thickness=-1)
        frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
        # Add text annotations
        for i, (label, value) in enumerate(label_value_dict.items()):
            color = bgr_to_rgb(getColor(i))
            text = (
                f"{label}: {value:.2f}"
                if isinstance(value, float)
                else f"{label}: {value}"
            )
            cv2.putText(
                frame_bgr,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        # !hahv: debug
        # if (vis_data_results is not None):
        #     pprint(vis_data_results['block_infos'] if 'block_infos' in vis_data_results else None)

        if vis_data_results and hasattr(self, "drawBlockDebug") and self.drawBlockDebug:
            if "topLeft" in vis_data_results and "rightBottom" in vis_data_results:
                # draw the ROI if available
                topLeft = vis_data_results["topLeft"]
                rightBottom = vis_data_results["rightBottom"]
                if isinstance(topLeft, tuple) and isinstance(rightBottom, tuple):
                    color_yellow = bgr_to_rgb((0, 255, 255))
                    cv2.rectangle(frame_bgr, topLeft, rightBottom, color_yellow, 2)
            if "block_infos" in vis_data_results:
                # draw debug blocks
                block_infos = vis_data_results["block_infos"]
                self._draw_block_debug(frame_bgr, block_infos)

        return frame_bgr

    def _pre_process_frame(self, frame, topleft_rightBottom_roi=None):
        """Pre-process the frame before inference.
        if roi is provided, it will crop the frame to the ROI.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_name = fs.get_file_name(self.cfg.model.model_path, split_file_ext=True)[0]
        pil_img = Image.fromarray(frame_rgb)
        if topleft_rightBottom_roi is not None:
            x0, y0 = topleft_rightBottom_roi[0]
            x1, y1 = topleft_rightBottom_roi[1]
            pil_img = pil_img.crop((x0, y0, x1, y1))

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
        # Apply the transformation
        frame_batch = transform(pil_img).unsqueeze(0)  # Add batch dimension
        # Move the frame batch to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame_batch = frame_batch.to(device)
        return frame_batch

    def _infer(self, frame_bsz1):  # frame_bsz1 is a batch of size 1
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        # move to device
        with torch.no_grad():
            logits = self.model(frame_bsz1)

        labelIdx = torch.argmax(logits, dim=1).item()
        # convert logits to numpy array
        logits = logits.cpu().numpy().squeeze()  # Remove batch dimension
        logits = logits.tolist()  # Convert to list for easier handling

        classNames = self.cfg.model.class_names
        assert labelIdx < len(classNames), "Class index out of range."
        pred_label = classNames[labelIdx]

        return logits, labelIdx, pred_label

    # # ! should be overridden by the subclass if needed
    # def custom_load_model(self):
    #     return super().custom_load_model()

    def before_video_infer(self, video_path):
        super().before_video_infer(video_path)
        if hasattr(self, "save_masks") and self.save_masks:
            self._init_mask_writer(video_path)

    # @abstractmethod
    # ! do_infer, infer_results, vis_data_results = self.detect_frame(...)
    # ! frame is from cv2.VideoCapture.
    @validate_detect_frame_return  # can disable this, if you want to skip validation
    def infer_frame(self, frame):
        frame_batch = None
        if self.method_name == "no_temp":
            shouldDoInfer = True  # always do inference
            frame_batch = self._pre_process_frame(frame, topleft_rightBottom_roi=None)
            vis_data_results = None
        else:  # "temp_stabilize"
            vis_data_results = None
            # default is no ROI (full frame)
            shouldDoInfer = False  # ! not always do inference
            topLeft, rightBottom = (0, 0), (frame.shape[1], frame.shape[0])
            maskFrame = None

            shouldDoInfer, (topLeft, rightBottom), vis_data_results, maskFrame = (
                self._tempStabilize(frame)
            )
            if shouldDoInfer:
                # Pre-process the frame
                frame_batch = self._pre_process_frame(frame, (topLeft, rightBottom))
                vis_data_results["topLeft"] = topLeft
                vis_data_results["rightBottom"] = rightBottom

            # if no infer, maskframe is just black
            if hasattr(self, "save_masks") and self.save_masks:
                if vis_data_results is None:
                    vis_data_results = {}
                vis_data_results["maskFrame"] = maskFrame

        # common operations
        logits, labelIdx, pred_label = None, None, None  # common init
        infer_result_dict = None
        # Perform inference if required
        if shouldDoInfer:
            logits, labelIdx, pred_label = self._infer(frame_batch)
            infer_result_dict = {
                "logits": logits,
                "pred_labelIdx": labelIdx,
                "pred_label": pred_label,
            }
        return shouldDoInfer, infer_result_dict, vis_data_results

    # ! abstractmethod
    def annotate_frame(
        self,
        frame,
        frame_idx,
        total_frames,
        infer_results=None,
        vis_data_results=None,
        gpu_stats=None,
    ):
        """Annotate the frame with detection results."""
        # pprint(infer_results)
        # frame is cv2.VideoCapture frame
        label_values_dict = {}
        label_values_dict["frameidx"] = f"{frame_idx + 1}/{total_frames}"
        if infer_results is not None:
            # infer_results should contain "outputs", "labelIdx", "className"
            logits = infer_results.get("logits")
            pred_labelIdx = infer_results.get("pred_labelIdx")
            pred_label = infer_results.get("pred_label")
            if (
                logits is not None
                and pred_labelIdx is not None
                and pred_label is not None
            ):
                probs = (
                    torch.nn.functional.softmax(torch.tensor(logits), dim=0)
                    .numpy()
                    .tolist()
                )
                probs_dict = dict(zip(self.cfg.model.class_names, probs))
                label_values_dict["--pred--"] = "-----"
                label_values_dict.update(probs_dict)
                label_values_dict["pred_label"] = pred_label

        # ! hahv: debug
        # pprint(gpu_stats)

        if gpu_stats:
            label_values_dict["---gpu--"] = "-----"
            if "gpu_avg_power" in gpu_stats:
                label_values_dict["gpu_power"] = f"{gpu_stats['gpu_avg_power']:.2f} W"

            if "gpu_avg_max_memory" in gpu_stats:
                label_values_dict["gpu_memory"] = (
                    f"{gpu_stats['gpu_avg_max_memory']:.2f} MB"
                )
        # pprint(label_values_dict)
        # pprint(f'{frame_idx=}, do_infer={infer_results is not None}, vis_data_results={vis_data_results is not None}')
        # pprint(vis_data_results)

        # in case we visualize mask
        if vis_data_results is not None:

            if "nonZeroCountTotal" in vis_data_results:
                label_values_dict["nonZeroCountTotal"] = vis_data_results[
                    "nonZeroCountTotal"
                ]
            if "nonZeroCountInside" in vis_data_results:
                label_values_dict["nonZeroCountInside"] = vis_data_results[
                    "nonZeroCountInside"
                ]

            if (
                "is_mask_frame" in vis_data_results
                and vis_data_results["is_mask_frame"] is True
            ):
                label_values_dict["--mask--"] = "-----"
                label_values_dict["heatMap"] = "Blue(low), Green(medium), Red(high)"

        return self._annotate_frame(
            frame,
            label_values_dict,
            vis_data_results=vis_data_results,
        )

    # ! abstractmethod
    def infer_results_to_list(self, infer_results):
        # csv_infer_cols = [<class_names>, <logits>, <probs>, <pred_label_idx>, <pred_label>]
        infer_row_data = []
        className = self.cfg.model.class_names
        logits = infer_results["logits"] if infer_results else None
        probs = (
            torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy().tolist()
            if logits
            else None
        )
        pred_labelIdx = infer_results["pred_labelIdx"] if infer_results else None
        pred_label = (
            className[pred_labelIdx]
            if className and pred_labelIdx is not None
            else None
        )
        infer_row_data.append(className)
        infer_row_data.append(logits)
        infer_row_data.append(probs)
        infer_row_data.append(pred_labelIdx)
        infer_row_data.append(pred_label)
        return infer_row_data

    # not required, so override if needed in this case we add info to masked
    def after_detect_frame(
        self,
        frame,
        frame_idx,
        total_frames,
        infer_results,
        vis_data_results,
        gpu_stats=None,
    ):
        if hasattr(self, "mask_writer") and self.mask_writer is not None:
            maskFrame = vis_data_results["maskFrame"]
            vis_data_results["is_mask_frame"] = True
            maskFrame = self.annotate_frame(
                maskFrame,
                frame_idx,
                total_frames,
                infer_results,
                vis_data_results,
                gpu_stats=gpu_stats,
            )
            if maskFrame is not None:
                # pprint('Writing mask frame to video...')
                self.mask_writer.write(maskFrame)

    def after_infer_video(self, video_path):
        super().after_video_infer(video_path)
        """Perform any necessary cleanup after inference ends."""
        if hasattr(self, "mask_writer") and self.mask_writer is not None:
            # Release the mask writer resource
            self.mask_writer.release()
            self.mask_writer = None
            with ConsoleLog("Masked saved"):
                pprint_local_path(self.out_mask_path)

    # ===========Metrics calculation methods================
    # ! override
    def get_exp_torch_metrics(self):
        """
        Return a dictionary of torchmetrics to be used for performance calculation.
        Example: {"accuracy": Accuracy(), "precision": Precision()}
        """
        return self.cfg.dataset.ds_metrics_dict

    # ! override
    def get_dataset_name(self):
        """
        Return the name of the dataset.
        This function should be overridden by the subclass if needed.
        """
        return self.cfg.dataset.target_ds_name

    # ! override
    def get_experiment_name(self):
        """
        Return the name of the experiment.
        This function should be overridden by the subclass if needed.
        """
        return self.cfg.method.selected_method

    def _csvfile_to_video_gt(self, csv_file):
        """
        Convert the CSV file to a video with ground truth labels.
        This function should be overridden by the subclass if needed.
        """
        if self.cfg.dataset.target_ds_name in ["DFire", "DFireStatic"]:
            fname = fs.get_file_name(csv_file, split_file_ext=True)[0]
            if "VP" in fname:
                return self.POSITIVE_LABEL
            elif "FP" in fname:
                return self.NEGATIVE_LABEL
            else:
                raise ValueError(
                    f"Unknown video ground truth label in file: {csv_file}"
                )
        elif self.cfg.dataset.target_ds_name == "FireNet":
            fname = fs.get_file_name(csv_file, split_file_ext=True)[0]
            if "NoFireVid" in fname:
                return self.NEGATIVE_LABEL
            else:
                return self.POSITIVE_LABEL
        elif self.cfg.dataset.target_ds_name == "pexels":
            return self.POSITIVE_LABEL # currently, all pexels videos are considered positive
        else:
            raise NotImplementedError(
                f"CSV file to video ground truth conversion is not implemented for dataset <<{self.cfg.dataset.target_ds_name}>>."
            )

    def _read_pred_csv(self, csv_file):
        """
        Read the prediction CSV file and return a DataFrame.
        This function should be overridden by the subclass if needed.
        """
        df = pd.read_csv(csv_file, sep=";")
        # if df["pred_label"] is NaN, convert it to "None" string
        df["pred_label"] = df["pred_label"].apply(
            lambda x: str(x) if not pd.isna(x) else "None"
        )
        return df

    def _run_tpt_method_on_df(self, df):
        """
        Run the temporal persistence thresholding (TPT) method on the DataFrame."""
        assert self.methodUsed.method_name == "temp_tpt", (
            f"Method {self.methodUsed.method_name} is not supported for this operation."
        )
        persistence_thres = self.persistence_thres
        pos = 0
        temporal_buffer = np.zeros(self.window_size, dtype=bool)
        detected = False
        for idx, row in df.iterrows():
            # Get the current prediction
            # pprint(row)
            pred_lb = row["pred_label"]
            pred_lb_lower = pred_lb.lower()
            is_fire_or_smoke = ('fire' in pred_lb_lower) or ('smoke' in pred_lb_lower)
            if is_fire_or_smoke:
                temporal_buffer[pos] = True
            pos = (idx + 1) % self.window_size # circular buffer
            # Check if the buffer has enough positive predictions
            num_det_frames = np.sum(temporal_buffer)
            if num_det_frames >= persistence_thres * self.window_size:
                detected = True
                break
        return self.POSITIVE_LABEL if detected else self.NEGATIVE_LABEL

    def _csvfile_to_pred_label(self, csv_file):
        """
        Convert the CSV file to a list of predicted labels.
        This function should be overridden by the subclass if needed.
        """
        if self.cfg.model.base_timm_model == "hgnetv2_b5.ssld_stage2_ft_in1k":
            NO_POSTPROC_METHODS = ['no_temp', 'temp_stabilize']
            if self.methodUsed.method_name in NO_POSTPROC_METHODS:
                # prof model
                # read that csv file:
                df = self._read_pred_csv(csv_file)
                preds = df["pred_label"].tolist()
                # check preds label values is nan, the convert to string "None"
                preds = [str(pred) if not pd.isna(pred) else "None" for pred in preds]
                # pprint(preds)
                preds_lowercase = [pred.lower() for pred in preds]
                # class_names = self.cfg.model.class_names
                # class_names: [Fire, None, SmokeOnly]
                found_fire_or_smoke = False
                for pred in preds_lowercase:
                    if ("fire" in pred) or ("smoke" in pred):
                        found_fire_or_smoke = True
                        break
                return self.POSITIVE_LABEL if found_fire_or_smoke else self.NEGATIVE_LABEL
            elif self.methodUsed.method_name == "temp_tpt":
                df = self._read_pred_csv(csv_file)
                # get tpt_pred for the video df
                tpt_pred_label = self._run_tpt_method_on_df(df)
                assert tpt_pred_label in [
                    self.POSITIVE_LABEL,
                    self.NEGATIVE_LABEL,
                ], f"Invalid tpt_pred label"
                return tpt_pred_label
            else:
                raise ValueError(
                    f"Unknown method name: {self.methodUsed.method_name}. Supported methods: {self.cfg.method.all_method_names}."
                )
        else:
            raise NotImplementedError(
                f"CSV file to predicted label conversion is not implemented for model {self.cfg.model.base_timm_model}."
            )

    def _data_from_metric(self, metric_mode, metric_name):
        """
        Normalize the data for metrics.
        This function should be overridden by the subclass if needed.
        """
        all_csv_files = fs.filter_files_by_extension(self.outdir, ".csv")
        all_csv_files = [
            file_item for file_item in all_csv_files if "__perf.csv" not in file_item
        ]  # ignore the performance csv file
        # pprint(f"Found {len(all_csv_files)} CSV files for metric '{metric_name}' in mode '{metric_mode}'.")

        RAW_DIR = "./zreport/raw"
        outdir_name = os.path.basename(self.outdir)
        out_raw_csvfile = os.path.join(RAW_DIR, f"{outdir_name}_raw.csv")
        saved_raw_csv = False
        if metric_mode == "per_video":
            if metric_name in [
                "accuracy",
                "precision",
                "recall (TPR)",
                "FPR",
                "f1_score",
            ]:
                preds = [
                    self._csvfile_to_pred_label(csv_file) for csv_file in all_csv_files
                ]
                # Convert preds to binary int torch tensor
                preds_tensor = [
                    1 if pred == self.POSITIVE_LABEL else 0 for pred in preds
                ]
                # Convert to torch tensor
                preds_tensor = torch.tensor(preds_tensor, dtype=torch.int)
                target = [
                    self._csvfile_to_video_gt(csv_file) for csv_file in all_csv_files
                ]
                # Convert target to binary int torch tensor
                target_tensor = [1 if t == self.POSITIVE_LABEL else 0 for t in target]
                target_tensor = torch.tensor(target_tensor, dtype=torch.int)

                if not saved_raw_csv:
                    dfmk = csvfile.DFCreator()
                    dfmk.create_table("raw_data", ["csv_outfile", "preds", "target"])
                    rows = []
                    for video_name, pred, tgt in zip(all_csv_files, preds, target):
                        rows.append([video_name, pred, tgt])

                    dfmk.insert_rows("raw_data", rows)
                    dfmk.fill_table_from_row_pool("raw_data")
                    df = dfmk['raw_data']
                    # new columns named "corrected?" and set to 1 if preds == target, else 0
                    df["corrected?"] = (df["preds"] == df["target"]).astype(int)
                    df = df.sort_values(by="corrected?", ascending=True)
                    df.to_csv(out_raw_csvfile, sep=";", index=False)
                    saved_raw_csv = True

                return {metric_name: {"preds": preds_tensor, "target": target_tensor}}
            elif metric_name == "FPS":
                elapsed_times_all_video = []
                for csv_file in all_csv_files:
                    single_video_df = pd.read_csv(csv_file, sep=";")
                    elapsed_times = single_video_df["elapsed_time"].tolist()
                elapsed_times_all_video.extend(elapsed_times)
                return {metric_name: elapsed_times_all_video}
            else:
                raise ValueError(
                    f"Metric '{metric_name}' is not supported in 'per_video' mode."
                )

        elif metric_mode == "per_frame":
            raise NotImplementedError(
                f"Metric mode '{metric_mode}' is not implemented for metric '{metric_name}'."
            )
        else:
            raise ValueError(
                f"Unknown metric mode: {metric_mode}. Supported modes: ['per_video', 'per_frame']"
            )

    # ! override
    def prepare_exp_data_for_metrics(self, metric_names, *args, **kwargs):
        """
        Prepare the data for metrics.
        This function should be overridden by the subclass if needed.
        Must return a dictionary with keys as metric names and values as the data to be used for those metrics.
        NOTE: that the data (for each metric) must be in the format expected by the torchmetrics instance (for that metric). E.g: {"accuracy": {"preds": [...], "target": [...]}, ...} since torchmetrics expects the data in a specific format.
        """

        metric_set = self.cfg.dataset.selected_metricSet
        METHOD_MODES = ["per_video", "per_frame"]
        metric_mode = metric_set.metric_set_cfgs.get("mode", "per_video")
        assert (
            metric_mode in METHOD_MODES
        ), f"Metric mode must be one of {METHOD_MODES}, got {metric_mode}."
        out_dict = {}
        for metric_name in metric_names:
            out_dict.update(self._data_from_metric(metric_mode, metric_name))
        return out_dict

    # ! override
    @valid_custom_fields
    def calc_exp_outdict_custom_fields(self, outdict, *args, **kwargs):
        """Can be overridden by the subclass to add custom fields to the output dictionary.
        ! must return the modified outdict, and a ordered list of custom fields to be added to the output dictionary.
        """
        method_params = {}
        custom_fields = []
        if hasattr(self, "methodUsed") and self.methodUsed is not None:
            if (
                not hasattr(self.methodUsed, "params")
                and self.methodUsed.params is not None
            ):
                method_params = (
                    self.methodUsed.params.copy()
                )  # make a copy of the params to avoid modifying the original
                if "method_name" in method_params:
                    del method_params["method_name"]  # remove method_name from params

        def flatten_dict(d, parent_key="", sep="."):
            items = {}
            for k, v in d.items():
                key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, key, sep=sep))
                else:
                    items[key] = v
            return items

        if len(method_params.keys()) > 0:
            params = flatten_dict(method_params)
            custom_fields = list(params.keys())
            for key, value in params.items():
                outdict[key] = value
        return outdict, custom_fields
