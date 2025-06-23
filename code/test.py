# ---------------------------
# 1. 모델 정의 (model.py)
# ---------------------------
from keras.models import Model
from keras.layers import Input
from keras.applications.xception import Xception as KerasXception

def build_xception_model():
    input_tensor = Input(shape=(299, 299, 3))
    model = KerasXception(weights=None, include_top=True, input_tensor=input_tensor, classes=1)
    return model


# ---------------------------
# 2. 얼굴 추출 함수 (preprocessing.py)
# ---------------------------
import cv2
import face_recognition

def extract_faces_from_video(video_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)

        for top, right, bottom, left in face_locations:
            face = frame[top:bottom, left:right]
            if face.size > 0:
                faces.append(face)

        frame_count += 1

    cap.release()
    return faces


# ---------------------------
# 3. 딥페이크 탐지 함수 (detector.py)
# ---------------------------
import numpy as np
import cv2
from model import build_xception_model
from preprocessing import extract_faces_from_video

def deepfake_detection(video_path, model_weights):
    print("[INFO] 비디오에서 얼굴 추출을 시작합니다...")
    faces = extract_faces_from_video(video_path)
    if not faces:
        print("[ERROR] 영상 내에서 얼굴을 찾지 못했습니다.")
        return None

    print(f"[INFO] 총 {len(faces)}개의 얼굴이 추출되었습니다.")

    print("[INFO] 모델 로딩 중...")
    model = build_xception_model()
    model.load_weights(model_weights)

    predictions = []
    for idx, face in enumerate(faces):
        try:
            face_processed = cv2.resize(face, (299, 299))
        except Exception as e:
            print(f"[WARNING] 얼굴 {idx+1} 리사이즈 중 오류 발생: {str(e)}")
            continue

        face_processed = face_processed.astype('float32') / 255.0
        face_processed = np.expand_dims(face_processed, axis=0)

        pred = model.predict(face_processed)
        probability = pred[0][0]
        predictions.append(probability)
        print(f"[INFO] 얼굴 {idx+1} - 딥페이크 확률: {probability:.4f}")

    if predictions:
        average_probability = np.mean(predictions)
        print(f"[RESULT] 전체 평균 딥페이크 확률: {average_probability:.4f}")
        return average_probability
    else:
        print("[ERROR] 예측 결과를 얻지 못했습니다.")
        return None


# ---------------------------
# 4. 메인 실행 함수 (main.py)
# ---------------------------
import os
import argparse
from detector import deepfake_detection

def main():
    parser = argparse.ArgumentParser(description="딥페이크 탐지 프로그램")
    parser.add_argument('--video', type=str, required=True, help="비디오 파일 경로")
    parser.add_argument('--weights', type=str, required=True, help="모델 가중치 파일 경로")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 비디오 파일을 찾을 수 없습니다: {args.video}")
        return
    if not os.path.exists(args.weights):
        print(f"[ERROR] 모델 가중치 파일을 찾을 수 없습니다: {args.weights}")
        return

    probability = deepfake_detection(args.video, args.weights)
    if probability is not None:
        print(f"[FINAL RESULT] 최종 딥페이크 판별 결과: {probability:.2f} (0: 진짜, 1: 딥페이크)")

if __name__ == '__main__':
    main()
