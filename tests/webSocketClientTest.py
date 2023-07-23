import asyncio
import websockets
import base64
import json


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
        # /connect에 연결 요청 메시지 전송
        connect_message = (
            "CONNECT\naccept-version:1.1,1.0\nheart-beat:10000,10000\n\n\0"
        )
        await websocket.send(connect_message)

        # 연결 응답 대기 (ACK 메시지 수신)
        response = await websocket.recv()
        print(f"서버에 연결됨: {response}")
        print("연결 완료!!!!")

        # /topic/messages 주제를 구독하는 메시지 전송
        subscribe_message = "SUBSCRIBE\nid:sub-0\ndestination:/topic/messages\n\n\0"
        await websocket.send(subscribe_message)
        print("/topic/messages uri 구독 완료")

        # 메시지 수신 대기 및 처리
        while True:
            message = await websocket.recv()
            message = message.split("\n")[-2].split(":")
            data = {message[0].split(" ")[4][1:-1]: message[1][2:-3]}

            # /app/receive 엔드포인트로 메시지 보내기
            send_message = (
                "SEND\ndestination:/app/receive\ncontent-type:application/json\n\n"
                + json.dumps(data)
                + "\0"
            )
            print("메세지 받음 : ", json.loads(json.dumps(data)))
            await websocket.send(send_message)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(client())
