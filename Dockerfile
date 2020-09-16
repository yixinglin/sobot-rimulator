FROM python:3

WORKDIR /usr/sobot-rimulator
COPY . .

RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
 && DEBIAN_FRONTEND="noninteractive" apt install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0

RUN pip3 install pycairo PyGObject
RUN pip3 install matplotlib numpy scipy
RUN pip3 install pyyaml

CMD [ "python3", "rimulator.py" ]