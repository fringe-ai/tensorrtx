import torch
import struct
import numpy as np
import time
from efficientnet_pytorch import EfficientNet

NUM_IMAGES = 3
DATA_TYPE = torch.half


if torch.cuda.is_available():
    print('using GPU')
    device = torch.device('cuda:0')
else:
    print('using CPU')
    device = torch.device('cpu')

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(device)
model.load_state_dict(torch.load('best.pt'))
model = model.to(DATA_TYPE)
model.eval()

proc_times = []
with torch.no_grad():
    dummy_input = torch.zeros(1,3,224,224).to(DATA_TYPE).to(device)
    for i in range(20):
        t1 = time.time()
        for _ in range(NUM_IMAGES):
            out = model(dummy_input)
        diff = time.time()-t1
        proc_times.append(diff*1000)
        print(f'proc time: {diff:.4f}')
proc_times = np.array(proc_times[1:])
print(out)
print(f'mean proc time: {proc_times.mean():.2f} ms')
print(f'max proc time: {proc_times.max():.2f} ms')
print(f'min proc time: {proc_times.min():.2f} ms')
