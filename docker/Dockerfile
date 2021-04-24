FROM python:3.7

ADD . /workspace/

WORKDIR /workspace

RUN apt update && apt-get install -y \
    python3 \
    python3-pip \
    libsnappy-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements-python.txt