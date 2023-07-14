from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# OpenCV의 Haar Cascade Classifier를 사용하여 얼굴 검출을 위한 분류기를 로드
face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
# 감정 분석을 위한 Keras 모델을 로드
emotion_model = load_model("./models/emotion_model.hdf5")

# 감정 레이블 정의
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def analyze_image(image):
    # 입력 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 얼굴 검출을 수행
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    for x, y, w, h in faces:
        # 얼굴 영역을 추출하고 사이즈를 조정합니다.
        roi_gray = gray_image[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        # 이미지를 정규화
        roi_gray = roi_gray / 255.0
        # Keras 모델에 입력하기 위해 차원을 확장
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # 감정 예측
        prediction = emotion_model.predict(roi_gray)
        probabilities = prediction[0]

        # 각각의 감정에 대한 확률을 딕셔너리로 생성
        labeled_probabilities = dict(zip(emotions, probabilities.tolist()))

        # 가장 높은 확률을 가진 감정을 결정
        max_index = np.argmax(probabilities)
        emotion = emotions[max_index]
        max_prob = probabilities[max_index]

        return {
            "emotion": emotion,
            "probability": labeled_probabilities,
            "stopVideo": emotion == "Happy" and max_prob > 0.5,
        }

    return {"error": "No face detected"}


@app.route("/")
def home():
    return render_template("index.html")  # 여기서 index.html은 당신의 홈페이지 템플릿 파일입니다.


@app.route("/analyze", methods=["POST"])
def analyze():
    # POST 요청에서 이미지 데이터를 가져옴.
    image = request.files["image"].read()
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
    # 이미지를 분석하여 결과를 반환
    result = analyze_image(image)

    if result.get("emotion") == "Happy" and result.get("stopVideo"):
        print("웃었네요.ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 종료합니다.")
        result["stopVideo"] = True

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
