# git clone https://github.com/ondyari/FaceForensics.git
# cd FaceForensics
# pip install -r requirements.txt

git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics
# 예시로 conda 환경 설정 (Python 3.7)
conda create -n ff_env python=3.7
conda activate ff_env
pip install -r requirements.txt


# 예시: 얼굴 추출 및 모델 예측
import cv2
import numpy as np
from model.classifier import Xception  # 저장소 내 제공된 모델 로딩 모듈
from utils.preprocessing import extract_faces  # 얼굴 추출 유틸리티

def deepfake_detection(video_path, model_weights):
    # 1. 영상에서 얼굴 영역을 추출 (프레임별로)
    faces = extract_faces(video_path)  # 영상 내 얼굴 이미지를 리스트로 반환
    if not faces:
        print("영상 내 얼굴을 찾지 못했습니다.")
        return

    # 2. 모델 로드 및 가중치 적용
    model = Xception()
    model.load_weights(model_weights)
    
    # 3. 각 얼굴 이미지에 대해 예측 실시 후 확률 평균 계산
    predictions = []
    for face in faces:
        # 모델에서는 입력 이미지 전처리가 필요할 수 있음 (예: 리사이즈, 정규화)
        face_processed = cv2.resize(face, (299, 299))  # Xception 입력 사이즈 예시
        face_processed = face_processed.astype('float32') / 255.0
        face_processed = np.expand_dims(face_processed, axis=0)
        pred = model.predict(face_processed)
        predictions.append(pred[0][0])  # 예시로 딥페이크 확률 반환

    average_probability = np.mean(predictions)
    return average_probability

if __name__ == '__main__':
    video_file = 'sample_video.mp4'
    model_path = 'weights/pretrained_model.h5'  # 사전 학습된 모델 파일 경로
    probability = deepfake_detection(video_file, model_path)
    print(f"딥페이크일 확률: {probability:.2f}")
