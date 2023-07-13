from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
emotion_model = load_model("./models/emotion_model.hdf5")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def analyze_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    for x, y, w, h in faces:
        roi_gray = gray_image[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = emotion_model.predict(roi_gray)
        probabilities = prediction[0]

        labeled_probabilities = dict(
            zip(emotions, probabilities.tolist())
        )  # Create a dictionary with emotions as keys and probabilities as values

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
    image = request.files["image"].read()
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
    result = analyze_image(image)

    if result.get("emotion") == "Happy" and result.get("stopVideo"):
        print("웃었네요. 종료합니다.")
        result["stopVideo"] = True

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
