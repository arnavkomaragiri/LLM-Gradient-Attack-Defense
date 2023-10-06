FROM nvcr.io/nvidia/pytorch:23.09-py3

ADD . /workspace/LLM-Gradient-Attack-Defense
WORKDIR /workspace/LLM-Gradient-Attack-Defense
RUN pip install -r requirements.txt
RUN /bin/bash
