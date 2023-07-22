from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from keras.models import load_model

from pydantic import BaseModel
import uvicorn
import json

# webSocket import
import asyncio
import websockets

app = FastAPI()


class AnalyzeResult(BaseModel):
    emotion: str
    probability: dict
    stopVideo: bool
    message: str


class testResult(BaseModel):
    message: str


async def client():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            image_data = await websocket.recv()
            result = await process_image(image_data)
            await websocket.send(json.dumps(result))


@app.get("/")
def home():
    return {"message": "모델 예측 서버 입장!!"}


@app.get("/test")
def test():
    return {"message": "/test에 대한 요청이 성공하였습니다!!!!!!!"}


@app.post("/analyze", response_model=testResult)
async def analyze(image_data: UploadFile = File(...)):
    return {"message": "하이"}
    print("여기까지옴")
    result = await process_image(image_data)
    return result


# 웹소켓 서버에서 전달받은 base64로 인코딩 된 데이터를 처리하는 함수
async def process_image(image_data: bytes):
    # image_data를 np배열로 변환
    np_array_image_data = image_to_array(image_data)

    # 모델을 통해 변환된 배열로 얼굴인식과 표정 분석 데이터를 BaseModel형식으로 결과를 받음
    return analyze_image(np_array_image_data)


def image_to_array(image_data):
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def analyze_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    for x, y, w, h in faces:
        roi_gray = gray_image[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)

        prediction = emotion_model.predict(roi_gray)
        probabilities = prediction[0]

        labeled_probabilities = dict(zip(emotions, probabilities.tolist()))

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


face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
emotion_model = load_model("./models/emotion_model.hdf5")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(client())
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
