import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# 사전 학습된 EfficientNet 모델 로드 (혹은 원하는 다른 모델 사용 가능)
model = models.efficientnet_b0(pretrained=True)
model.eval()

# 전처리 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 딥페이크 탐지 함수
def detect_deepfake(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.sigmoid(output).item()
    
    return prediction  # 0에 가까우면 실제, 1에 가까우면 딥페이크

# 사용자에게 영상 파일 받기
video_path = input("딥페이크를 판별할 영상 파일 경로를 입력하세요: ")

# 영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 프레임을 한 장씩 읽어들이면서 딥페이크 판별
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 딥페이크 가능성 계산
    score = detect_deepfake(frame)
    label = "Deepfake" if score > 0.5 else "Real"

    # 결과 표시
    cv2.putText(frame, f"Prediction: {label} ({score:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Deepfake Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
