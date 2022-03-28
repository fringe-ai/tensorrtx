model_path='./trained-inference-models/2022-03-28_1024_256/best.pt'
num_classes=2
test_imgs='./data/cropped_binary_1024_256'
build_path='./build_nordson'
output_path='./validation/2022-03-28_1024_256'
model='b0'

mkdir -p "$output_path"

if [ ! -f "$output_path"/efficientnet-"$model".wts ]; then 
    python gen_wts.py -w "$model_path" -c "$num_classes" -o "$output_path"/efficientnet-"$model".wts
else
    echo 'wts already exists, skip'
fi

if [ ! -f "$output_path"/efficientnet-"$model".engine ]; then 
    echo "$build_path"/efficientnet -s "$output_path"/efficientnet-"$model".wts "$output_path"/efficientnet-"$model".engine "$model"
    "$build_path"/efficientnet -s "$output_path"/efficientnet-"$model".wts "$output_path"/efficientnet-"$model".engine "$model"
else
    echo 'engine already exists, skip'
fi

python run_inference.py -e "$output_path"/efficientnet-"$model".engine -i "$test_imgs"
