FROM tensorflow/serving:latest

COPY models/iris /models/iris

ENV MODEL_NAME=iris

EXPOSE 8501