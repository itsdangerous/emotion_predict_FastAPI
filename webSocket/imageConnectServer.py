import asyncio
import websockets
import base64
import json

import cv2
import numpy as np

from keras.models import load_model
from imageDataToNp import image_data_to_np
from analyzingImage import analyze_image

face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
emotion_model = load_model("./models/emotion_model.hdf5")

# 감정 레이블
emotions = ["Angry", "Disgust", "Fear", "Smile", "Sad", "Surprise", "Neutral"]


async def connect_to_server(websocket):
    # /connect에 연결 요청 메시지 전송
    connect_message = "CONNECT\naccept-version:1.1,1.0\nheart-beat:10000,10000\n\n\0"
    await websocket.send(connect_message)

    # 연결 응답 대기 (ACK 메시지 수신)
    response = await websocket.recv()
    print(f"서버에 연결됨: {response}")
    print("연결 완료!!!!")

    # /topic/messages 주제를 구독하는 메시지 전송
    subscribe_message = "SUBSCRIBE\nid:sub-0\ndestination:/topic/imageData\n\n\0"
    await websocket.send(subscribe_message)
    print("/topic/imageData uri 구독 완료")

async def receive_and_process_message(websocket):
    while True:
        # 메시지 수신 대기 및 처리
        message = await websocket.recv()  # 웹소켓 서버로부터 메시지 받음
        body = await extract_body(message)  # 메시지의 헤더와 바디를 분리하여 바디만 추출
        processed_image = await image_data_to_np(body)  # 분리된 body를 np배열로 변환
        emotion_result = await analyze_image(  # np배열로 변환된 데이터로, 표정분석
            processed_image, face_cascade, emotion_model, emotions
        )

        print(type(emotion_result))

        # 분석 결과를 JSON 형태로 변환
        emotion_result_json = json.dumps(emotion_result, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)

        print(emotion_result_json)

        # /app/analyze 엔드포인트로 분석 결과 전송
        analyze_message = f"SEND\ndestination:/app/analyzingData\n\n{emotion_result_json}\0"
        await websocket.send(analyze_message)


async def extract_body(message):
    # 먼저 bytes를 문자열로 변환
    message = message.decode("utf-8")

    # 두 개의 연속된 개행 문자를 기준으로 헤더와 본문을 분리
    headers, body = message.split("\n\n", 1)

    # 본문을 다시 bytes로 변환
    return body.encode()


async def client():
    uri = "ws://localhost:8080/ws"  # 자바 웹소켓 서버의 주소

    # Spring boot Security 사용자 이름
    username = "1234"

    # Spring Boot Security 비밀번호
    password = "1234"
    encoded_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

    # 웹소켓 서버에 연결
    async with websockets.connect(
        uri, extra_headers={"Authorization": f"Basic {encoded_credentials}"}
    ) as websocket:
        await connect_to_server(websocket)
        await receive_and_process_message(websocket)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(client())
