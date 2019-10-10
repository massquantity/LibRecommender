#!/bin/sh

sudo docker run -p 8501:8501 --mount type=bind,source=$(pwd)/models/FM,target=/models/FM -e MODEL_NAME=FM -t tensorflow/serving



