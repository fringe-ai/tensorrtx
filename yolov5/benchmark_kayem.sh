#!/bin/bash

#Activate the yolo environments, you may want to modify the repo paths
source ~/projects/venv_pipeline/bin/activate
source ~/projects/yolov5/yolo.env

#----------------------------------------------------------
# arguments
DATA_PATH='./data/kayem_split_640_0.2/test'
BUILD_PATH='./build_kayem'
CLASSES='filling,sausage,underfill,cojoin,misform'
WEIGHTS_PATH='./trained-inference-models/2022-02-17'
PROJECT_NAME='2022-02-17'
MODELS='s m l x'
OUTPUT_PATH='./validation'
#----------------------------------------------------------

for MODEL in $MODELS
do
    #intermediate folder names
    model_name="$PROJECT_NAME"_"$MODEL"
    out_folder="$OUTPUT_PATH"/"$model_name"
    out_name="$out_folder"/"$model_name"
    echo
    echo $out_name

    #create folder
    mkdir -p "$out_folder"

    #Genrate the weights
    if [ ! -f "$out_name".wts ]; then
        echo "generating the weights file"
        python gen_wts.py -w "$WEIGHTS_PATH"/"$model_name".pt -o "$out_name".wts
    else
        echo "found the file: ""$out_name".wts
        echo "skip"
    fi

    #Build engine
    if [ ! -f "$out_name".engine ]; then
        "$BUILD_PATH"/yolov5 -s "$out_name".wts "$out_name".engine "$MODEL"
    else
        echo "found the file: ""$out_name".engine
        echo "skip"
    fi

    #RUN inference
    python run_inference.py -e "$out_name".engine -i "$DATA_PATH" -p "$BUILD_PATH" -c "$CLASSES" -o "$out_folder"
done
