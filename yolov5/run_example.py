import os
import subprocess
import yaml


data_path = './data/nordson_2022-03-24_split_0.2/test'
build_path = './build'
config_file = './configs/example.yaml'
classes = 'scuff,white,peeling'
weights_path = './trained-inference-models/2022-03-24'
model_name = '2022-03-24_1024_256_600'
output_path = './validation'
yolo_repo_path = '$HOME/projects/yolov5'
#-------------------------------------------------------------------

with open(config_file) as f:
    dt = yaml.safe_load(f)
    model = dt['YOLO']['model']
    print(f'loaded YOLO model from yaml: {model}')
if f'{model}'!=model_name.split('_')[-1]:
    model_name += f'_{model}'
    print(f'append model {model} to the model name: {model_name}')
    
out_folder = os.path.join(output_path, model_name)
out_name = os.path.join(out_folder, model_name)

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

# Genrate the weights
wts_file = out_name+'.wts'
if not os.path.isfile(wts_file):
    print('generating the weights file')
    os.environ['PYTHONPATH'] = os.path.expandvars(yolo_repo_path)
    weight_file = os.path.join(weights_path, model_name+'.pt')
    cmd = ['python3', '-m', 'gen_wts', '-w', weight_file, '-o', wts_file]
    r = subprocess.run(cmd, env=dict(os.environ))
    r.check_returncode()
else:
    print(f'found wts file: {wts_file}')

# Build engine
engine_file = out_name+'.engine'
if not os.path.isfile(engine_file):
    cmd = [build_path+'/yolov5', '-c', config_file, '-w', wts_file, '-o', engine_file]
    r = subprocess.run(cmd)
    r.check_returncode()
else:
    print(f'found engine file: {engine_file}')
    
# Run inference
cmd = ['python3', 'run_inference.py', '-e', engine_file, 
       '-i', data_path, '-p', build_path, '-c', classes, '-o', out_folder]
r = subprocess.run(cmd)
r.check_returncode()
