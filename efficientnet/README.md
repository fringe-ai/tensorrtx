# EfficientNet

A TensorRT implementation of EfficientNet.
For the Pytorch implementation, you can refer to the **YJ** branch of [EfficientNet-PyTorch](https://github.com/fringe-ai/EfficientNet-PyTorch.git)

## How to run

1. install `efficientnet_pytorch`
```
pip install efficientnet_pytorch
```

2. gennerate `.wts` file for building the tensorRT engine \
The command below assumes that the pytorch weights `best.pt` exists in current directory, and the model predicts 5 classes. It will generate the output file as `./efficientnet-b0_2.wts`.
```
python gen_wts.py -w ./best.pt -c 5 -o ./efficientnet-b0.wts 
```

3. **modify the settings in `efficientnet.cpp`** 
```c++
#define USE_FP16 //or USE_FP32
#define MAX_BATCH_SIZE 1
```

Change the number of classes in the corresponding model. Below, it changes the number of classes in *b0* model to 5. 
```c++
static std::map<std::string, GlobalParams>
	global_params_map = {
		// input_h,input_w,num_classes,batch_norm_epsilon,
		// width_coefficient,depth_coefficient,depth_divisor, min_depth
		{"b0", GlobalParams{224, 224, 5, 0.001, 1.0, 1.0, 8, -1}}, //change to 5 classes
		{"b1", GlobalParams{240, 240, 1000, 0.001, 1.0, 1.1, 8, -1}},
		{"b2", GlobalParams{260, 260, 1000, 0.001, 1.1, 1.2, 8, -1}},
		{"b3", GlobalParams{300, 300, 1000, 0.001, 1.2, 1.4, 8, -1}},
		{"b4", GlobalParams{380, 380, 1000, 0.001, 1.4, 1.8, 8, -1}},
		{"b5", GlobalParams{456, 456, 1000, 0.001, 1.6, 2.2, 8, -1}},
		{"b6", GlobalParams{528, 528, 1000, 0.001, 1.8, 2.6, 8, -1}},
		{"b7", GlobalParams{600, 600, 1000, 0.001, 2.0, 3.1, 8, -1}},
		{"b8", GlobalParams{672, 672, 1000, 0.001, 2.2, 3.6, 8, -1}},
		{"l2", GlobalParams{800, 800, 1000, 0.001, 4.3, 5.3, 8, -1}},
};
```


4. compile the C++ project

```
mkdir build
cd build
cmake ..
make
```


5. serialize model to engine
```
./efficientnet -s [.wts] [.engine] [b0 b1 b2 b3 ... b7]  // serialize model to engine file
```
such as
```
./efficientnet -s ../efficientnet-b0.wts efficientnet-b0.engine b0
```

6. deserialize and do infer using tensorRT C++ API (**depreciated**)
```
./efficientnet -d [.engine] [b0 b1 b2 b3 ... b7]   // deserialize engine file and run inference
```
such as 
```
./efficientnet -d efficientnet-b0.engine b0
```

7. run inference using tensorRT Python API \
-e: the engine file \
-i: the testing image path
```
python ../run_inference.py -e ./efficientnet-b0.engine -i ../data/cropped_224x224
```


For more models, please refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
