# Setting

<!-- OS | Ubuntu 18.04 LTS in Windows WSL1 -->

OS | Window 10
Language |Python 3.11.4
Package Manager | pip 23.2
Async Web Framework | FastAPI 0.68.2
WSGI | gunicorn 20.1.0

<!-- ASGI | uvicorn 0.15.0 -->

model Framework | tensorflow 2.12.0

requirements.txt is served

## Follow

pip install --upgradae pip -> pip-23.2 <br>
python -m venv .venv <br>

<!-- . ./venv/bin/activate <br> -->

source /venv/Scripts/activate
pip install -r requirements.txt <br>

<!-- sudo ./.venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:80 <br> -->

sudo ./.venv/Scripts/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:80 <br>

### run server

api swagger :: localhost/docs <br>


<br><br>

connectServer.py라는 파일을 하나 더 만들었습니다.<br>
이 파일은 자바 서버와 웹통신을 위해, 파이썬 서버가 클라이언트 역할을 하는 코드입니다.

실행하려면 아래 명령을 실행하면 됩니다.<br>
python connectServer.py
