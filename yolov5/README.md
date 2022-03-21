# yolov5 implementation

The Pytorch implementation is `${fringe-ai}/yolov5`: https://github.com/fringe-ai/yolov5.git.

## Train and Test on your own datasets
https://github.com/fringe-ai/yolov5/blob/YJ/TRAIN_AND_TEST.md


## Config

- Choose the model n/s/m/l/x/n6/s6/m6/l6/x6 from command line arguments.
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in yolov5.cpp, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp
- Batch size in yolov5.cpp


## How to Run, yolov5s with your own model

1. generate .wts from pytorch with .pt

```bash
#clone the {tensorrtx}/yolov5 repo:
git clone https://github.com/lmitechnologies/tensorrtx.git

#clone the {fringe-ai}/yolov5 repo:
git clone https://github.com/lmitechnologies/yolov5.git

#copy pytorch weights (best.pt) to fringe-ai yolov5 repo
gsutil -m cp gs://engagements/nordson/chattanooga/catheter/feasibility/models/pytorch/defeat/objdet/yolov5/training/2022-01-05_640/weights/best.pt {fringe-ai}/yolov5

#copy the gen_wts.py to fringe-ai yolov5 repo
cp {tensorrtx}/yolov5/gen_wts.py {fringe-ai}/yolov5

#generate a .wts file
cd {fringe-ai}/yolov5
python gen_wts.py -w best.pt -o 2022-01-05_640.wts
```

2. modify the configs

- modify the following configs in `yololayer.h`
```c++
static constexpr int CLASS_NUM = 3;
static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
static constexpr int INPUT_W = 640;
```

- modify the following configs in `yolov5.cpp`
```c++
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
```

3. build tensorrtx/yolov5 and generate tensorRT engine
```bash
# build tensorrtx/yolov5
cd {tensorrtx}/yolov5/
mkdir build
cd build
cmake ..
make

#serialize model and weights to build the tensorRT engine
cp {fringe-ai}/yolov5/2022-01-05_640.wts .
./yolov5 -s 2022-01-05_640.wts 2022-01-05_640.engine s
```

4. run the inference
```bash
cd {tensorrtx}/yolov5
python run_inference.py -e ./build/2022-01-05_640.engine -i ./data/2022-01-04_640 -c peeling,scuff,white -o ./validation/2022-01-05_640
```


# INT8 Quantization

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

