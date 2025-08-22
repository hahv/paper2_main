from halib import *
import line_profiler
import torch
import timm
import cv2
import time
import numpy as np
from torchvision import transforms
from PIL import Image

base_timm_model = "hgnetv2_b5.ssld_stage2_ft_in1k"
model_path = "models/prof_hgnetv2_b5.ssld_stage2_ft_in1k-20250620-50-98.78.pth"
class_names = [
    "Fire",
    "Normal",
    "SmokeOnly",
]
video_source = r"dataset/smallfire.mp4"
# video_source = r"dataset/wrong_fd_cam_stream.mp4"
# video_source = r"dataset/f6.mp4"

def load_model(base_timm_model, class_names, model_path):
    model = timm.create_model(
        base_timm_model,
        pretrained=False,
        num_classes=len(class_names),
        checkpoint_path=model_path,
    )
    model.eval()  # evaluation mode
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


@line_profiler.profile
def main():
    global base_timm_model, class_names, model_path, video_source
    # Load model + transforms
    model = load_model(base_timm_model, class_names, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((360, 640)),  # keep ratio fixed for inference
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(video_source)

    NUM_FRAME_TRIAL = 99999
    frame_idx = 0
    time_ls = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        start_time = time.time()

        # Preprocess frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_class = torch.max(probs, 0)


        # !Measure elapsed time
        elapsed = time.time() - start_time
        time_ls.append(elapsed)

        # Overlay prediction on frame
        label = f"{class_names[pred_class]} ({conf.item()*100:.1f}%)"
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"{label} | FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        SCALE = 0.5
        frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Prediction", frame)

        # stop after NUM_FRAME_TRIAL or ESC
        if frame_idx >= NUM_FRAME_TRIAL:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # drop first frame (cold start) timing
    if len(time_ls) > 1:
        time_ls = time_ls[1:]

    avg_fps = 1.0 / np.mean(time_ls) if len(time_ls) > 0 else 0
    print(f"Processed {frame_idx} frames. Avg FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    pprint("Starting inference profiling...")
    main()
