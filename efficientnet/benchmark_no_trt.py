import torch
import struct
import numpy as np
import time
from efficientnet_pytorch import EfficientNet

NUM_IMAGES = 3

if torch.cuda.is_available():
    print('using GPU')
    device = torch.device('cuda:0')
else:
    print('using CPU')
    device = torch.device('cpu')

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(torch.double).to(device)
model.load_state_dict(torch.load('best.pt'))
model.eval()

dummy_input = np.ones((1,3,224,224))*0.1
dummy_input = torch.from_numpy(dummy_input).to(torch.double).to(device)
proc_times = []
for i in range(20):
    start_time = time.time()
    for _ in range(NUM_IMAGES):
        out = model(dummy_input)
    proc_time = time.time()-start_time
    print(f'{proc_time:.4f}')
    proc_times.append(proc_time)
proc_times = np.array(proc_times[1:])
print(out)
print(f'proc time: {proc_times.mean()*1000:.2f} ms')
