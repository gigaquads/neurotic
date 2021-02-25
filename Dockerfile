FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONPATH=/src
WORKDIR /src
COPY . /tmp
WORKDIR /tmp
RUN pip3 install --upgrade pip
RUN pip3 install cython numpy pandas sklearn tabulate
WORKDIR /src
