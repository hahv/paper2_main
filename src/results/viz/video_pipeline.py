from halib import *
import cv2
from typing import List
from src.results.viz.base_renderer import BaseRenderer


class VideoPipeline:
    def __init__(self, filepath: str, fps: float, frame_size: tuple):
        self.filepath = filepath
        self.frame_size = frame_size  # ! must be (width, height)
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
        self.writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
        self.renderers: List[BaseRenderer] = []

    def add_renderer(self, renderer: BaseRenderer):
        self.renderers.append(renderer)

    def process_and_write(self, raw_frame, global_context: dict):
        """Passes frame through all renderers, then writes to disk."""
        if not self.writer.isOpened():
            console.print(
                f"[red]Video writer not opened for file: {self.filepath}[/red]"
            )
            return

        frame = raw_frame.copy()
        for renderer in self.renderers:
            renderer_ctx = renderer.global_ctx_to_render_ctx(global_context)
            frame = renderer.render(frame, renderer_ctx)
        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()
            print(f"Video saved: {self.filepath}")
