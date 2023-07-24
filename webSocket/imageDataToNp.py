import cv2
import numpy as np
import base64


async def image_data_to_np(image: bytes):
    try:
        # bytes를 문자열로 변환
        image_str = image.decode("utf-8")

        # 'data:' 이후의 데이터를 추출
        image_str = image_str.split(",", 1)[1]

        # base64 디코딩
        image_data = base64.b64decode(image_str)

        # bytes를 numpy 배열로 변환
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # 이미지 데이터로 변환 후 반환
        result = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
        if result is None:
            print("cv2.imdecode failed.")
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
