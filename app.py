from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# from typing import List
import cv2
import numpy as np
from keras.models import load_model
from pydantic import BaseModel
import uvicorn

# webSocket connect
import asyncio
import websockets

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    while True:
        try:
            async with websockets.connect('ws://localhost:8080/ws') as websocket:
                print("자바 웹소켓 서버에 연결을 성공하였습니다.")
                data = await websocket.recv()
                # 이렇게 받은 데이터를 이용해 원하는 작업을 수행하시면 됩니다.
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK):
            print("연결이 끊어졌습니다... 5초 동안 재연결을 시도합니다.")
            await asyncio.sleep(5)

# 각 분석 결과를 정의하기 위한 Pydantic 모델
class AnalyzeResult(BaseModel):
    emotion: str
    probability: dict
    stopVideo: bool
    message: str

@app.get("/")
def home() :
    return {"message" : "모델 예측 서버 입장!!"}

@app.get("/test")
def test():
    return {"message": "/test에 대한 요청이 성공하였습니다!!!!!!!"}


@app.get("/video", response_class=HTMLResponse)
async def video(request: Request):
    # FastAPI에서는 별도로 HTML 템플릿을 처리하는 기능이 없으므로, video.html을 반환하는 대신 사용자에게 해당 메시지를 보여줍니다.
    return templates.TemplateResponse("video.html", {"request": request})



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
    uvicorn.run("app:app", host="0.0.0.0", port=80, reload=True)