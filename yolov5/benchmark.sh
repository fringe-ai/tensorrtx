#!/bin/bash

#Activate the yolo environments, you may want to modify the repo paths
source ~/projects/venv_pipeline/bin/activate
source ~/projects/yolov5/yolo.env

#----------------------------------------------------------
# arguments
DATA_PATH='./data/split_640_0.2/test'
CLASSES='filling,sausage,underfill,cojoin,misform'
WEIGHTS_PATH='./trained-inference-models/2022-02-17'
PROJECT_NAME='2022-02-17'
MODELS='s m l x'
#----------------------------------------------------------

for MODEL in $MODELS
do
    #intermediate folder names
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
done
