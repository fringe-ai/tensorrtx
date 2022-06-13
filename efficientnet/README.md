# EfficientNet

A TensorRT implementation of EfficientNet.
For the Pytorch implementation, you can refer to [EfficientNet-PyTorch](https://github.com/fringe-ai/EfficientNet-PyTorch.git)

## How to run

1. install `efficientnet_pytorch`
```
pip install efficientnet_pytorch
```

2. gennerate `.wts` file for building the tensorRT engine \
The command below assumes that the pytorch weights `best.pt` exists in current directory, and the model predicts 2 classes. It will generate the output file as `./efficientnet-b0.wts`.
```
python gen_wts.py -w ./best.pt -c 2 -o ./efficientnet-b0.wts 
```

3. compile the C++ project

```
mkdir build
cd build
cmake ..
make
```


4. **modify the settings in `configs/example.yaml`** 
```yaml
EFFICIENT_NET:
  backbone: b0  # backbones: [b0 b1 b2 b3 ... b7]
  input_h: 512
  input_w: 512
  num_classes: 2
  batch_size: 1
```


5. serialize model to engine
```
./efficientnet -c [.yaml] -w [.wts] -o [.engine]  // serialize model to engine file
```
such as
```
./efficientnet -c ./configs/example.yaml -w efficientnet-b0.wts -o efficientnet-b0.engine 
```


6. run inference using tensorRT Python API \
-e: the engine file \
-i: the testing image path
```
python ./run_inference.py -e ./efficientnet-b0.engine -i ./data
```


For more models, please refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
