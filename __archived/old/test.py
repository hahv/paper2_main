from halib import *
from PIL import Image
from _archived.config_builder import ConfigFactory
from _archived._firedet import FireDetector

factory = ConfigFactory("./config/_base.yaml", "./config/temporal")
cfg = factory.build()

pprint(str(cfg.expConfig.list_methods["method1"]))

detector = FireDetector(cfg)
img = Image.open(r"./test/images/21deutsche.span.jpg").convert("RGB")
detector.loadModel()
video_path = r"./test/videos/f6.mp4"
with timebudget("Inference time"):
    # Run inference
    # result = detector._infer(img)
    detector.infer_video(video_path)
# pprint(result)
