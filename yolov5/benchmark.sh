#!/bin/bash

#NOTE: Activate the python virtual environment FIRST

#Activate the yolo environment, you may want to modify the yolov5 repo path
source ~/projects/yolov5/yolo.env

#----------------------------------------------------------
# arguments
DATA_PATH='./data/pad_to_640'
CLASSES='sausage,cojoin,underfill,filling,misform'
WEIGHTS_PATH='./trained-inference-models/2022-02-13'
PROJECT_NAME='2022-02-13'
MODEL='x'
#----------------------------------------------------------

OUTPUT_PATH='./validation'
OUT_NAME="$PROJECT_NAME"_"$MODEL"

#Genrate the weights
if [ ! -f "$OUT_NAME".wts ]; then
    echo "generating the weights file"
    python gen_wts.py -w "$WEIGHTS_PATH"/"$OUT_NAME".pt -o "$OUT_NAME".wts
else
    echo "found the file: ""$OUT_NAME".wts
    echo "skip"
fi

#Build engine
if [ ! -f "$OUT_NAME".engine ]; then
    ./build/yolov5 -s "$OUT_NAME".wts "$OUT_NAME".engine "$MODEL"
else
    echo "found the file: ""$OUT_NAME".engine
    echo "skip"
fi

#RUN inference
python run_inference.py -e "$OUT_NAME".engine -i "$DATA_PATH" -c "$CLASSES" -o "$OUTPUT_PATH"/"$OUT_NAME"
