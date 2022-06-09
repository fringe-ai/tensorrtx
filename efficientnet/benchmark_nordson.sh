model_path='./trained-inference-models/2022-04-25_512_512/best.pt'
config_path='./configs/example.yaml'
build_path='./build'
test_imgs='./data/cropped_binary_512x512'
output_path='./validation/2022-04-25_512_512'
model='b0'
num_classes=2
#-----------------------------------------------------------------------------------------------------

source ~/projects/venv_pipeline/bin/activate

mkdir -p "$output_path"

if [ ! -f "$output_path"/efficientnet-"$model".wts ]; then 
    python gen_wts.py -w "$model_path" -c "$num_classes" -o "$output_path"/efficientnet-"$model".wts
else
    echo 'wts already exists, skip'
fi

if [ ! -f "$output_path"/efficientnet-"$model".engine ]; then 
    "$build_path"/efficientnet -c $config_path -w "$output_path"/efficientnet-"$model".wts -o "$output_path"/efficientnet-"$model".engine
else
   echo 'engine already exists, skip'
fi

python run_inference.py -e "$output_path"/efficientnet-"$model".engine -i "$test_imgs"
