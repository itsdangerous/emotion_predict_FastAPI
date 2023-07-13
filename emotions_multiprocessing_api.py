from flask import Flask, Response, render_template
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from multiprocessing import Process, Queue

app = Flask(__name__)

# 얼굴 감지 XML 로드 및 훈련된 모델 로드
face_detection = cv2.CascadeClassifier("models//haarcascade_frontalface_default.xml")
emotion_classifier = load_model("models/emotion_model.hdf5", compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Smile", "Sad", "Surprising", "Neutral"]

# 감정 색상 정의
EMOTION_COLORS = {
    "Angry": (255, 0, 0),
    "Disgusting": (0, 255, 0),
    "Fearful": (0, 0, 128),
    "Smile": (0, 0, 255),
    "Sad": (128, 0, 0),
    "Surprising": (0, 255, 255),
    "Neutral": (255, 255, 255),
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


def generate_frames(camera):
    face_queue = Queue()
    result_queue = Queue()

    frame_counter = 0  # 프레임 카운터 초기화

    while True:
        # 이미지 캡처
        frame = camera.read()[1]

        # 얼굴 감지 결과를 받아오기 위한 큐 비우기
        while not face_queue.empty():
            face_queue.get()

        # 프로세스 생성 및 시작
        process = Process(
            target=process_frame,
            args=(frame, face_queue, result_queue, frame_counter),
        )
        process.start()

        # 프로세스 종료 대기
        process.join()

        # 얼굴 감지 결과 가져오기
        faces = face_queue.get()

        # 결과 큐에서 결과 가져오기
        result = result_queue.get()

        # 결과를 프레임에 그리기
        if len(faces) > 0:
            (fX, fY, fW, fH) = result[0]
            # 라벨링 지정
            color = EMOTION_COLORS[result[1]]
            cv2.putText(
                frame,
                result[1],
                (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), color, 2)

        # 이미지를 바이트 스트림으로 인코딩
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # 바이트 스트림 반환
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        frame_counter += 1


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    camera = cv2.VideoCapture(0)
    return Response(
        generate_frames(camera), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
