import torch
import struct
import numpy as np
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).to(torch.double)
model.load_state_dict(torch.load('best.pt'))
model.eval()

dummy_input = np.ones((1,3,224,224))*0.1
dummy_input = torch.from_numpy(dummy_input).to(torch.double)
out = model(dummy_input)
print(out)


f = open('efficientnet-b0.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
f.close()
