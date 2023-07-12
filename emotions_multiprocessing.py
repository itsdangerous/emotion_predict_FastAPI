import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from multiprocessing import Process, Queue

# 얼굴 감지 XML 로드 및 훈련된 모델 로드
face_detection = cv2.CascadeClassifier("models//haarcascade_frontalface_default.xml")
emotion_classifier = load_model("models/emotion_model.hdf5", compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Smile", "Sad", "Surprising", "Neutral"]

# 감정 색상 정의
EMOTION_COLORS = {
    "Angry": (255, 0, 0),  # 파랑
    "Disgusting": (0, 255, 0),  # 초록
    "Fearful": (0, 0, 128),  # 진한 빨강
    "Smile": (0, 0, 255),  # 빨강
    "Sad": (128, 0, 0),  # 진한 파랑
    "Surprising": (0, 255, 255),  # 시안
    "Neutral": (255, 255, 255),  # 흰색
}

# 감정 분석 빈도
analyze_frequency = 10  # 10 프레임마다 감정 분석


def process_frame(frame, face_queue, result_queue, frame_counter):
    # 색상을 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 프레임 내의 얼굴 감지
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 얼굴이 감지될 때만 감정 인식 수행
    if len(faces) > 0:
        # 가장 큰 이미지에 대해
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[
            0
        ]
        (fX, fY, fW, fH) = face
        # 신경망을 위해 이미지 크기를 64*64로 조정
        roi = gray[fY : fY + fH, fX : fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 'analyze_frequency' 프레임마다 감정 예측
        if frame_counter % analyze_frequency == 0:
            preds = emotion_classifier.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]

        # 결과 큐에 결과 전달
        result_queue.put((face, label))

    # 얼굴 큐에 얼굴 위치 전달
    face_queue.put(faces)


def display_frames(face_queue, result_queue):
    # 웹캠 개수
    num_cameras = face_queue.qsize()

    # 웹캠 비디오 캡처
    cameras = [cv2.VideoCapture(i) for i in range(num_cameras)]

    frame_counter = 0  # 프레임 카운터 초기화

    while True:
        # 각 웹캠에서 이미지 캡처
        frames = [camera.read()[1] for camera in cameras]

        # 얼굴 감지 결과를 받아오기 위한 큐 비우기
        while not face_queue.empty():
            face_queue.get()

        # 프로세스 생성 및 시작
        processes = [
            Process(
                target=process_frame,
                args=(frame, face_queue, result_queue, frame_counter),
            )
            for frame in frames
        ]
        for process in processes:
            process.start()

        # 프로세스 종료 대기
        for process in processes:
            process.join()

        # 얼굴 감지 결과 가져오기
        faces = [face_queue.get() for _ in range(num_cameras)]

        # 결과 큐에서 결과 가져오기
        results = [result_queue.get() for _ in range(num_cameras)]

        # 결과를 프레임에 그리기
        for i, (frame, (face, label)) in enumerate(zip(frames, results)):
            if len(faces[i]) > 0:
                (fX, fY, fW, fH) = face
                # 라벨링 지정
                color = EMOTION_COLORS[label]
                cv2.putText(
                    frame,
                    label,
                    (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), color, 2)

        # 이미지 표시 ("Emotion Recognition")
        if len(frames) > 0:
            cv2.imshow("Emotion Recognition", np.hstack(frames))

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_counter += 1  # 프레임 카운터 증가

    # 프로그램 정리 및 창 닫기
    for camera in cameras:
        camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 웹캠 개수
    num_cameras = 2

    # 얼굴 큐 및 결과 큐 생성
    face_queue = Queue()
    result_queue = Queue()

    # 멀티프로세스로 웹캠 프레임 처리
    p = Process(target=display_frames, args=(face_queue, result_queue))
    p.start()

    # 메인 프로세스에서 멀티프로세스 종료 대기
    p.join()
