from temporal.methods.base_method import *
from temporal.methods.no_temp import NoTempMethod
from torchvision import transforms
from PIL import Image
from temporal.config import Config
from temporal.metric_src.test_metric_src import TestDSMetricSrc
from temporal.tiny_cnn import *
import pybgs as bgs
import xml.etree.ElementTree as ET


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

    def do_infer(self, model, input_tensor):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
        return preds

    def extract_blocks_torch(self, scaled_frame_bgr, block_size):
        # frame_bgr: (H,W,3) uint8 numpy
        frame = torch.from_numpy(scaled_frame_bgr).permute(2, 0, 1)  # (C,H,W), still uint8
        frame = frame[[2, 1, 0], :, :]  # BGR → RGB
        frame = frame.float() / 255.0  # normalize in torch
        # unfold extracts sliding blocks, then reshape to grid
        blocks = frame.unfold(1, block_size, block_size).unfold(
            2, block_size, block_size
        )
        blocks = blocks.contiguous().view(3, -1, block_size, block_size)
        return blocks.permute(1, 0, 2, 3)  # (num_blocks,C,block_size,block_size)

    def _active_blocks(self, fg_mask, threshold=0.1):
        """
        fg_mask: (H,W) uint8 numpy (0 or 255)
        returns: 1D array of active block indices
        """
        H, W = fg_mask.shape
        blk_h, blk_w = H // self.blk_size, W // self.blk_size

        # reshape into blocks (avoid copies)
        blocks = fg_mask[:blk_h*self.blk_size, :blk_w*self.blk_size].reshape(
            blk_h, self.blk_size, blk_w, self.blk_size
        ).swapaxes(1, 2)  # (blk_h, blk_w, blk_size, blk_size)

        # count active pixels in each block
        # 255 -> 1
        counts = (blocks > 0).sum(axis=(2,3))
        total_pixels = self.blk_size * self.blk_size

        # boolean mask of active blocks
        active_mask = counts / total_pixels > threshold

        # convert to 1D indices (row-major order)
        active_indices = np.flatnonzero(active_mask)
        return active_indices, blk_h, blk_w

    def skip_module(self, cv2_bgr_frame, extract_func, scale_factor=1):
        # pprint('Running skip module...')
        # pprint(f"Skip module scale factor: {scale_factor}")
        should_skip = False
        roi_rect = None
        o_H, o_W = cv2_bgr_frame.shape[:2]
        minW = int(self.min_roi * o_W)
        minH = int(self.min_roi * o_H)
        #! skip module using SCALED FRAME (not original frame) for speed up
        if scale_factor < 1.0:
            cv2_bgr_frame = cv2.resize(
                cv2_bgr_frame,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA,
            )
        scaled_H, scaled_W = cv2_bgr_frame.shape[:2]
        # Calculate padding to make dimensions divisible by blk_size
        pad_h = (self.blk_size - (scaled_H % self.blk_size)) % self.blk_size
        pad_w = (self.blk_size - (scaled_W % self.blk_size)) % self.blk_size

        # Pad the frame if necessary
        if pad_h > 0 or pad_w > 0:
            cv2_bgr_frame = cv2.copyMakeBorder(
                cv2_bgr_frame,
                top=0,
                bottom=pad_h,
                left=0,
                right=pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 255, 0),  # Green padding
            )

        fg_mask = self.algorithm.apply(cv2_bgr_frame)
        print(f'Frame max value in fg mask: {fg_mask.max()}')
        # 2. Get active blocks
        active_indices, blk_h, blk_w = self._active_blocks(fg_mask, threshold=0.1)
        print(f"Number of active blocks: {len(active_indices)}")

        max_num_blocks = blk_h * blk_w
        pprint(f"Max number of blocks: {max_num_blocks}")
        active_percent = len(active_indices) / max_num_blocks

        # 3. Make sure we have active blocks and not too many active blocks (exceed threshold)
        if len(active_indices) > 0 and active_percent < self.frm_act_thres:
            blocks_cpu = extract_func(cv2_bgr_frame, block_size=self.blk_size)
            blocks_active = blocks_cpu[active_indices]
            preds = self.do_infer(self.tiny_block_model, blocks_active.to(self.device))
            preds = preds.cpu().numpy()
            # only keep active blocks predicted as fire/smoke
            firesmoke_active_indices = active_indices[preds == 1]
            # Convert indices to rows and columns in the block grid
            rows = firesmoke_active_indices // blk_w
            cols = firesmoke_active_indices % blk_w

            # Block coordinates in scaled frame
            x = cols * self.blk_size
            y = rows * self.blk_size
            w = h = self.blk_size  # scalar

            # Map back to original frame and clip to frame boundaries
            x_orig = np.minimum((x / scale_factor).astype(int), o_W - 1)
            y_orig = np.minimum((y / scale_factor).astype(int), o_H - 1)
            w_orig = np.minimum(int(w / scale_factor), o_W - x_orig)
            h_orig = np.minimum(int(h / scale_factor), o_H - y_orig)

            # Combine into array of coordinates
            coords_original = np.stack([x_orig, y_orig, w_orig, h_orig], axis=1)  # shape: (num_blocks, 4)
            x1 = coords_original[:, 0].min()
            y1 = coords_original[:, 1].min()
            x2 = (coords_original[:, 0] + coords_original[:, 2]).max()
            y2 = (coords_original[:, 1] + coords_original[:, 3]).max()
            # roi_rect = (x1, y1, x2, y2)
            # Ensure minimum ROI size
            roi_w = x2 - x1
            roi_h = y2 - y1

            if roi_w < minW:
                # expand equally left and right, clip to frame
                expand_w = (minW - roi_w) // 2
                x1 = max(0, x1 - expand_w)
                x2 = min(o_W, x2 + expand_w)
            if roi_h < minH:
                # expand equally top and bottom, clip to frame
                expand_h = (minH - roi_h) // 2
                y1 = max(0, y1 - expand_h)
                y2 = min(o_H, y2 + expand_h)

            roi_rect = (x1, y1, x2, y2)
        else:
            # ! too many active blocks, do not skip, make ROI whole frame
            roi_rect = None # ! if ROI is None, means whole frame
            should_skip = False
            return should_skip, roi_rect, fg_mask

        roi_rect = None
        should_skip = False
        return should_skip, roi_rect, fg_mask

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        should_skip = False
        should_skip, _, fg_mask = self.skip_module(
            frame,
            extract_func=self.extract_blocks_torch,
            scale_factor=self.scale_factor,
        )
        if not should_skip:
            return super().infer_frame(frame, frame_idx)
        else:
            # pprint(f"Frame {frame_idx} skipped by skip module.")
            return {
                "logits": [0.0] * len(self.cfg.model_cfg.class_names),
                "probs": [0.0] * len(self.cfg.model_cfg.class_names),
                "predLabelIdx": -1,
                "predLabel": "skipped",
            }
