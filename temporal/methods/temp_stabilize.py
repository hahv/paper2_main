from temporal.methods.base_method import *
from temporal.methods.no_temp import NoTempMethod
from temporal.tiny_cnn import *
import pybgs as bgs
import xml.etree.ElementTree as ET
import torch.nn.functional as F


class TempStabilizeMethod(NoTempMethod):
    def update_threshold(self, file_path, new_value):
        tree = ET.parse(file_path)
        root = tree.getroot()

        # find the <threshold> tag and update its value
        threshold_elem = root.find("threshold")
        if threshold_elem is not None:
            threshold_elem.text = str(new_value)
            tree.write(file_path, encoding="utf-8", xml_declaration=True)
            print(f"Updated <threshold> to {new_value} in {file_path}")
        else:
            print("<threshold> tag not found in the XML.")

    def _load_temp_stabilize_cfg(self):
        method_dict = self.cfg.method_cfg.method_used.extra_cfgs
        with ConsoleLog("temp_steablize cfg:"):
            pprint(method_dict)
        self.diff_thres = method_dict["diff_thres"]
        self.blk_size = method_dict["blk_size"]
        self.blk_act_thres = method_dict["blk_act_thres"]
        self.frm_act_thres = method_dict["frm_act_thres"]
        self.fire_cls_thres = method_dict["fire_cls_thres"]
        self.min_roi = method_dict["min_roi"]
        self.tiny_model_path = method_dict["tiny_model"]
        frame_diff_cfg = method_dict["frame_diff_cfg"]
        self.scale_factor = method_dict["scale_factor"]
        self.update_threshold(frame_diff_cfg, self.diff_thres)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_block_tiny_cnn_model(self):
        assert self.tiny_model_path is not None, "Tiny model path is not set."
        assert os.path.isfile(self.tiny_model_path), (
            f"Tiny model file does not exist: {self.tiny_model_path}"
        )
        self.tiny_block_model = TinyCNN(num_classes=2)
        self.tiny_block_model.load_state_dict(
            torch.load(self.tiny_model_path, map_location=self.device)
        )
        self.tiny_block_model.to(self.device)
        self.tiny_block_model.eval()

    def before_infer_video_dir(self, video_dir: str):
        assert self.cfg.method_cfg.method_used.name == "temp_stabilize", (
            f"Method {self.cfg.method_cfg.method_used.name} is not supported for this operation"
        )
        self._load_temp_stabilize_cfg()
        self._load_block_tiny_cnn_model()

    def before_infer_video(self, video_path: str):
        self.algorithm = bgs.FrameDifference()  # background subtraction algorithm

    def do_tinycnn_infer(self, model, input_tensor):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        return preds, probs

    def extract_blocks_torch(self, scaled_frame_bgr, block_size):
        # frame_bgr: (H,W,3) uint8 numpy
        frame = torch.from_numpy(scaled_frame_bgr).permute(
            2, 0, 1
        )  # (C,H,W), still uint8
        frame = frame[[2, 1, 0], :, :]  # BGR → RGB
        frame = frame.float() / 255.0  # normalize in torch
        # unfold extracts sliding blocks, then reshape to grid
        blocks = frame.unfold(1, block_size, block_size).unfold(
            2, block_size, block_size
        )
        blocks = blocks.contiguous().view(3, -1, block_size, block_size)
        return blocks.permute(1, 0, 2, 3)  # (num_blocks,C,block_size,block_size)

    def _active_motion_blocks(self, fg_mask, threshold=0.1):
        """
        fg_mask: (H,W) uint8 numpy (0 or 255)
        returns: 1D array of active block indices
        """
        H, W = fg_mask.shape
        blk_h, blk_w = H // self.blk_size, W // self.blk_size

        # reshape into blocks (avoid copies)
        blocks = (
            fg_mask[: blk_h * self.blk_size, : blk_w * self.blk_size]
            .reshape(blk_h, self.blk_size, blk_w, self.blk_size)
            .swapaxes(1, 2)
        )  # (blk_h, blk_w, blk_size, blk_size)

        # count active pixels in each block
        # 255 -> 1
        counts = (blocks > 0).sum(axis=(2, 3))
        total_pixels = self.blk_size * self.blk_size

        # percentage of active pixels in each block
        percentages = counts / total_pixels
        # boolean mask of active blocks
        active_mask = percentages > threshold

        # convert to 1D indices (row-major order)
        active_indices = np.flatnonzero(active_mask)
        active_percentages = percentages.flatten()[active_indices]
        return active_indices, active_percentages, blk_h, blk_w

    def _calculate_roi_from_indices(
        self, firesmoke_active_indices, blk_w, scale_factor, orig_shape
    ):
        """
        Calculates the final ROI in original frame coordinates from active block indices.
        """
        o_H, o_W = orig_shape
        assert self.min_roi > 0 and self.min_roi <= 1.0, "min_roi must be in (0,1]"
        (min_o_H, min_o_W) = int(self.min_roi * o_H), int(self.min_roi * o_W)

        # Step 1: Convert 1D indices to 2D block grid coordinates (rows, cols) using vectorized operations
        rows, cols = np.divmod(firesmoke_active_indices, blk_w)

        # Step 2: Calculate top-left and bottom-right corners in SCALED frame
        x1_scaled = cols * self.blk_size
        y1_scaled = rows * self.blk_size
        x2_scaled = x1_scaled + self.blk_size
        y2_scaled = y1_scaled + self.blk_size

        # Step 3: Compute tight bounding box in SCALED frame
        min_x, min_y = np.min(x1_scaled), np.min(y1_scaled)
        max_x, max_y = np.max(x2_scaled), np.max(y2_scaled)

        # Step 4: Map to ORIGINAL frame coordinates
        box = np.array([min_x, min_y, max_x, max_y]) / scale_factor
        x1, y1, x2, y2 = box.astype(int)

        # Step 5: Enforce minimum ROI size
        roi_w, roi_h = x2 - x1, y2 - y1
        x1 -= (min_o_W - roi_w) // 2 if roi_w < min_o_W else 0
        x2 += (min_o_W - roi_w) // 2 if roi_w < min_o_W else 0
        y1 -= (min_o_H - roi_h) // 2 if roi_h < min_o_H else 0
        y2 += (min_o_H - roi_h) // 2 if roi_h < min_o_H else 0

        # Step 6: Clip to original frame boundaries
        return (max(0, x1), max(0, y1), min(o_W, x2), min(o_H, y2))

    def _resize_and_pad(self, frame, scale_factor):
        if scale_factor < 1.0:
            scaled_frame_bgr = cv2.resize(
                frame,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA,
            )
        else:
            scaled_frame_bgr = frame

        scaled_H, scaled_W = scaled_frame_bgr.shape[:2]
        pad_h = (self.blk_size - (scaled_H % self.blk_size)) % self.blk_size
        pad_w = (self.blk_size - (scaled_W % self.blk_size)) % self.blk_size

        if pad_h > 0 or pad_w > 0:
            scaled_frame_bgr = cv2.copyMakeBorder(
                scaled_frame_bgr,
                0,
                pad_h,
                0,
                pad_w,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return scaled_frame_bgr

    def skip_module(self, frame_idx, cv2_bgr_frame, extract_func, scale_factor=1):
        """
        Analyzes a frame to decide whether to skip it or process a specific ROI.

        This optimized version fixes a critical bug, removes redundant code,
        and improves performance by reusing the efficient _calculate_roi_from_indices method.
        """
        o_H, o_W = cv2_bgr_frame.shape[:2]

        # Step 1: Scale and pad the frame for block processing
        # This part remains the same as it uses efficient OpenCV functions.
        self.profiler.step_start(ctx_name="skip_module", step_name="resize_and_pad")
        scaled_frame_bgr = self._resize_and_pad(cv2_bgr_frame, scale_factor)
        self.profiler.step_end(ctx_name="skip_module", step_name="resize_and_pad")

        # Step 2.1: Identify blocks with motion using the foreground mask
        self.profiler.step_start(ctx_name="skip_module", step_name="fg_mask")
        fg_mask = self.algorithm.apply(scaled_frame_bgr)

        # ! this is for visualization
        fg_mask_dict = {
            "fg_mask": fg_mask,
            "block_size": self.blk_size,
            "active_motion_blocks_info": [],
            "firesmoke_blocks_cls_info": {
                "all_active_blocks": [],
                "firesmoke_active_blocks": [],
            },
            "ROI_rect": None,
            "active_percent": 0.0,
            "global_info": {
                "o_shape": (o_H, o_W),
                "scale_factor": scale_factor,
            },
        }

        self.profiler.step_end(ctx_name="skip_module", step_name="fg_mask")
        # Step 2.2: Determine active blocks based on the foreground mask
        self.profiler.step_start(ctx_name="skip_module", step_name="active_blocks")
        active_indices, active_percentages,blk_h, blk_w = self._active_motion_blocks(
            fg_mask, threshold=self.blk_act_thres
        )
        # ! udate fg_mask_dict
        fg_mask_dict["active_motion_blocks_info"] = list(zip(active_indices, active_percentages))

        self.profiler.step_end(ctx_name="skip_module", step_name="active_blocks")

        max_num_blocks = blk_h * blk_w
        # Step 3: Decide action based on the amount of motion
        # Case A: No motion detected -> SKIP the frame
        if len(active_indices) == 0:
            console.print(f"[{frame_idx}] No motion detected. Skipping frame.")
            return True, None, fg_mask_dict

        active_percent = len(active_indices) / max_num_blocks
        fg_mask_dict["active_percent"] = active_percent
        console.print(
            f"[{frame_idx}] Active motion blocks: [red]{len(active_indices)}[/red]/{max_num_blocks} ({active_percent:.2%})"
        )

        # Case B: Too much motion (e.g., camera pan) -> PROCESS the WHOLE frame
        if active_percent >= self.frm_act_thres:
            console.print(
                f"[{frame_idx}] Motion exceeds threshold. Processing full frame."
            )
            roi_rect = (0, 0, o_W, o_H)
            return False, roi_rect, fg_mask_dict

        # Case C: Reasonable motion -> Run classifier on active blocks
        self.profiler.step_start(ctx_name="skip_module", step_name="firesmoke_active_blocks")

        blocks_cpu = extract_func(scaled_frame_bgr, block_size=self.blk_size)
        blocks_active = blocks_cpu[active_indices]
        preds, probs = self.do_tinycnn_infer(
            self.tiny_block_model, blocks_active.to(self.device)
        )
        preds = preds.cpu().numpy()
        probs = probs.cpu().numpy()   # shape: (num_active_blocks, num_classes)

        # mask of blocks predicted as fire_smoke
        firesmoke_mask = preds == TinyCNN.FIRE_SMOKE_CLASS_IDX
        firesmoke_active_indices = active_indices[firesmoke_mask]

        # raw probabilities (percentages) for each active block
        firesmoke_probs = probs[:, TinyCNN.FIRE_SMOKE_CLASS_IDX]   # take probability of fire_smoke
        firesmoke_active_probs = firesmoke_probs[firesmoke_mask]

        # ! update fg_mask_dict
        # vis all active blocks and their probs
        fg_mask_dict["firesmoke_blocks_cls_info"]["all_active_blocks"] = list(zip(active_indices, firesmoke_probs))

        # vis only blocks classified as fire/smoke
        fg_mask_dict["firesmoke_blocks_cls_info"]["firesmoke_active_blocks"] = firesmoke_active_indices

        console.print(
            f"[{frame_idx}] Blocks classified as fire/smoke: [green]{len(firesmoke_active_indices)}[/green]/{len(active_indices)}"
        )

        self.profiler.step_end(ctx_name="skip_module", step_name="firesmoke_active_blocks")

        # Subcase C1: Motion was found, but none was classified as fire/smoke -> SKIP the frame
        if len(firesmoke_active_indices) == 0:
            return True, None, fg_mask_dict

        # Subcase C2: Fire/smoke blocks found -> Calculate a tight ROI and PROCESS it
        self.profiler.step_start(ctx_name="skip_module", step_name="calc_roi")
        roi_rect = self._calculate_roi_from_indices(
            firesmoke_active_indices, blk_w, scale_factor, (o_H, o_W)
        )
        self.profiler.step_end(ctx_name="skip_module", step_name="calc_roi")
        return False, roi_rect, fg_mask_dict

    def after_infer_video_dir(self, video_dir):
        self.profiler.save_report_dict(output_file=f"{self.cfg.get_outdir()}/profiler_report.json", with_detail=True)
        self.profiler.report_and_plot(self.cfg.get_outdir())

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        self.profiler.ctx_start(ctx_name="infer_frame")
        self.profiler.step_start(ctx_name="infer_frame", step_name="skip_module")
        self.profiler.ctx_start(ctx_name="skip_module")

        should_skip, roi_rect, fg_mask_dict = self.skip_module(
            frame_idx,
            frame,
            extract_func=self.extract_blocks_torch,
            scale_factor=self.scale_factor,
        )
        fg_mask_dict["ROI_rect"] = roi_rect
        self.profiler.ctx_end("skip_module")
        self.profiler.step_end(ctx_name="infer_frame", step_name="skip_module")
        if roi_rect is not None:
            pprint(f"[{frame_idx}] ROI: {roi_rect}")
        # ! force no skip for testing
        # should_skip = False
        if not should_skip:
            self.profiler.step_start(ctx_name="infer_frame", step_name="big_infer")
            res = super().infer_frame(frame, frame_idx)
            self.profiler.step_end(ctx_name="infer_frame", step_name="big_infer")
            self.profiler.ctx_end("infer_frame")
            # add fg_mask_dict to res
            res["fg_mask_dict"] = fg_mask_dict
            return res
        else:
            # pprint(f"Frame {frame_idx} skipped by skip module.")
            res = {
                "logits": [0.0] * len(self.cfg.model_cfg.class_names),
                "probs": [0.0] * len(self.cfg.model_cfg.class_names),
                "predLabelIdx": -1,
                "predLabel": "skipped",
            }
            res["fg_mask_dict"] = fg_mask_dict
            return res
