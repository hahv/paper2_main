import torch
from torchvision import transforms
from PIL import Image
import timm

# 모델 경로와 클래스 수 설정
MODEL_PATH = "pretrained/hgnetv2_b5.ssld_stage2_ft_in1k-20250601-96-98.80.pth.tar"
NUM_CLASSES = 3  # 클래스 수를 실제 모델에 맞게 수정
BACKBONE = "hgnetv2_b5.ssld_stage2_ft_in1k"

# 모델 로드
model = timm.create_model(BACKBONE, pretrained=False, num_classes=NUM_CLASSES, checkpoint_path=MODEL_PATH)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((360, 640)), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 이미지 불러오기
img = Image.open('0135.png').convert('RGB')
input_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가

# 추론
with torch.no_grad():
    output = model(input_tensor)    

# 결과 출력
output = output.argmax(dim=1).item()  # 클래스 인덱스 추출    
print(output) #0: fire, 1: none, 2: smoke