# Setting
OS | Ubuntu 18.04 LTS in Windows WSL1
Language |Python 3.11.4
Package Manager | pip 23.2
Async Web Framework | FastAPI 0.68.2
WSGI | gunicorn 20.1.0
ASGI | uvicorn 0.15.0
model Framework | tensorflow 2.12.0

## requirements.txt is served

### follow

pip install --upgradae pip -> pip-23.2
python -m venv .venv
. ./venv/bin/activate
pip install -r requirements.txt
sudo ./.venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:80

api swagger :: localhost/docs