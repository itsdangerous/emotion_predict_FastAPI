import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image

# 얼굴 감지 XML 로드
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


# 표정 분류 모델 정의 및 로드
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(48 * 48, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = x.view(-1, 48 * 48)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
model.load_state_dict(torch.load("models/emotion_model.pt"))
model.eval()

# 표정 클래스 레이블
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 웹캠을 사용한 비디오 캡처
cap = cv2.VideoCapture(0)

while True:
    # 카메라에서 이미지 캡처
    ret, frame = cap.read()

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 감지
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 얼굴이 감지되면 표정 예측 수행
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # 얼굴 영역 추출
        face_roi = gray[y : y + h, x : x + w]
        # 얼굴 영역 크기 조정
        face_roi = cv2.resize(face_roi, (48, 48))
        # PIL 이미지로 변환
        pil_image = Image.fromarray(face_roi)

        # 이미지를 텐서로 변환
        image_tensor = ToTensor()(pil_image).unsqueeze(0)

        # 텐서를 모델에 전달하여 예측 수행
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_emotion = emotion_labels[predicted.item()]

        # 얼굴 주위에 사각형과 예측 결과 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            predicted_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # 결과를 출력
    cv2.imshow("Facial Expression Recognition", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 프로그램 정리 및 창 닫기
cap.release()
cv2.destroyAllWindows()
