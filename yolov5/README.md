# yolov5 Pytorch to TensorRT 

The Pytorch implementation is `${lmi}/yolov5`: https://github.com/lmitechnologies/yolov5.

## Train and Test on your own datasets
https://github.com/lmitechnologies/yolov5/blob/master/TRAIN_AND_TEST.md


## How to convert trained yolo model to tensorRT
The followings assume that the current working directory is `~/projects`, and the trained yolo weights are saved as `~/projects/best.pt`.

1. Clone the source code repos

The following example clones the repos to `~/repos`.
```bash
# cd to the directory where you want to clone the repos
cd ~/repos

# clone the tensorrtx repo:
git clone git@github.com:lmitechnologies/tensorrtx.git

# clone the LMI_AI_SOLUTIONS repo:
git clone git@github.com:lmitechnologies/LMI_AI_Solutions.git
```

2. Generate `example.wts` from pytorch weights `best.pt`

```bash
# activate the lmi_ai env
source PATH_TO_REPO/LMI_AI_Solutions/lmi_ai.env

# generate a .wts file
python3 -m yolov5.gen_wts -w best.pt -o example.wts
```

3. Modify the yaml config

Use [configs/example.yaml](https://github.com/lmitechnologies/tensorrtx/blob/master/yolov5/configs/example.yaml) as a template for creating your own config file. The followings show you what is inside that yaml.
```yaml
YOLO:
  model: s # model types: [n/s/m/l/x/n6/s6/m6/l6/x6] or [c/c6 gd gw]
  input_h: 256
  input_w: 1024
  num_classes: 3
  batch_size: 1
```
Create your own config file and save it as `~/projects/example.yaml`.

4. build and generate tensorRT engine
```bash
mkdir build
cd build
cmake ~/repos/tensorrtx/yolov5
make

#build the tensorRT engine
cd ..
./build/yolov5 '-c' example.yaml '-w' example.wts '-o' ./build/out.engine
```
The engine is saved as `./build/out.engine`.

5. run the inference
```bash
python3 ~/repos/tensorrtx/yolov5/run_inference.py -e ./build/out.engine -p ./build -i DATA_PATH -c CLASS_NAMES -o DESTINATION
```
where `DATA_PATH` is the path to the image folder, `CLASS_NAMES` is a string containing all class names separated by a comma, `DESTINATION` is the output path.

# INT8 Quantization (haven't tested)

1. Prepare calibration images, you can randomly select 1000s images from your train set. For coco, you can also download my calibration images `coco_calib` from [GoogleDrive](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing) or [BaiduPan](https://pan.baidu.com/s/1GOm_-JobpyLMAqZWCDUhKg) pwd: a9wh

2. unzip it in yolov5/build

3. set the macro `USE_INT8` in yolov5.cpp and make

4. serialize the model and test

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247927-4d9fac00-751e-11ea-8b1b-704a0aeb3fcf.jpg">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/78247970-60b27c00-751e-11ea-88df-41473fed4823.jpg">
</p>

## More Information

See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)

