import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 사전 학습된 딥러닝 모델 로드
model = load_model('model/deepfake_cnn.h5')

# 영상에서 프레임 추출
def extract_frames(video_path, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        frame_count += 1
    cap.release()
    return np.array(frames) / 255.0

# 딥페이크 여부 분석
def analyze_video(video_path):
    frames = extract_frames(video_path)
    if len(frames) == 0:
        return "분석 실패: 영상 프레임 없음"
    preds = model.predict(frames)
    score = np.mean(preds)
    return "딥페이크" if score > 0.5 else "진짜 영상"
