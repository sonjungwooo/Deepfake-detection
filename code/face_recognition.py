import cv2
import face_recognition

def extract_faces(video_path, max_frames=50):
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
