from halib import *
from PIL import Image
from _archived.config_builder import ConfigFactory
from temporal.methods.temp_stabilize import ProposedDetector

factory = ConfigFactory("./config/_base.yaml", "./config/methods")
cfg = factory.build()
pprint(cfg)
detector = ProposedDetector(cfg)
# video_path = r"./test/videos/f6.mp4"
video_path = r"E:\NextCloud\paper3\datasets\FireNet\NoFire\NoFireVid4.mp4"
# video_path = r"test\videos\FP42.mp4"
with timebudget("Inference time"):
    detector.infer_video(video_path)
