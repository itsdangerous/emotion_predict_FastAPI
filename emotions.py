import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

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
frame_counter = 0
last_label = None

# 웹캠을 사용한 비디오 캡쳐
camera = cv2.VideoCapture(0)

while True:
    # 카메라에서 이미지 캡쳐
    ret, frame = camera.read()
    # 색상을 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 프레임 내의 얼굴 감지
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 빈 이미지 생성
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # 얼굴이 감지될 때만 감정 인식 수행
    if len(faces) > 0:
        # 가장 큰 이미지에 대해
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
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
            last_label = EMOTIONS[preds.argmax()]

        # 라벨링 지정
        color = EMOTION_COLORS[last_label]
        cv2.putText(
            frame, last_label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
        )
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), color, 2)

        # 라벨 출력
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(
                canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1
            )
            cv2.putText(
                canvas,
                text,
                (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )

    # 두 개의 창 열기
    ## 이미지 표시 ("Emotion Recognition")
    ## 감정 확률 표시
    cv2.imshow("Emotion Recognition", frame)
    cv2.imshow("Probabilities", canvas)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_counter += 1  # 프레임 카운터 증가

# 프로그램 정리 및 창 닫기
camera.release()
cv2.destroyAllWindows()
