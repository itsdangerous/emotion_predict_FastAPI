from fastapi import FastAPI, UploadFile, File

# from typing import List
import cv2
import numpy as np
from keras.models import load_model
from pydantic import BaseModel
import uvicorn

app = FastAPI()


# 각 분석 결과를 정의하기 위한 Pydantic 모델
class AnalyzeResult(BaseModel):
    emotion: str
    probability: dict
    stopVideo: bool
    message: str


@app.get("/")
def home():
    return {"message": "모델 예측 서버 입장!!"}


@app.get("/test")
def test():
    return {"message": "/test에 대한 요청이 성공하였습니다!!!!!!!"}


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(image: UploadFile = File(...)):
    result = await process_image(image)
    print(result)
    return result


async def process_image(image: UploadFile):
    # 이미지를 numpy 배열로 변환
    result = image_to_array(await image.read())

    # 이미지를 분석하고 결과 반환
    return analyze_image(result)


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
            "message": "success",
        }

    return {
        "emotion": "Not Detected Your Face",
        "probability": dict(zip(emotions, [-1] * 7)),
        "stopVideo": False,
        "message": "failed",
    }


def image_to_array(image_data):
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


# 모델 로드는 일회성 작업이므로 전역 영역에서 한 번만 수행
face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
emotion_model = load_model("./models/emotion_model.hdf5")

# 감정 레이블 정의
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True, Debug=True)
