"""
An example that uses TensorRT's Python api to make inferences.
"""
from logging import raiseExceptions
import os
import random
import time
import cv2
from skimage.io import imread
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt



def get_img_path_batches(batch_size, img_dir):
    import json
    ret = []
    ret_label = []
    batch = []
    batch_label = []

    #load class map
    class_to_id = {}
    class_map_path = os.path.join(img_dir,'class_map.json')
    if os.path.isfile(class_map_path):
        class_to_id = json.load(open(class_map_path))
    else:
        raise Exception(f'cannot load the class_map.json in {img_dir}')

    # load images with class id
    for root, dirs, _files in os.walk(img_dir):
        for dir in dirs:
            label = class_to_id[dir]
            for file in os.listdir(os.path.join(root, dir)):
                if len(batch) == batch_size:
                    ret.append(batch)
                    ret_label.append(batch_label)
                    batch = []
                    batch_label = []
                batch.append(os.path.join(root, dir, file))
                batch_label.append(label)
    if len(batch) > 0:
        ret.append(batch)
        ret_label.append(batch_label)
    return ret, ret_label

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class EfficientNetTRT(object):
    """
    description: A EfficientNet class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.batch_size = self.engine.max_batch_size
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            print('bingding:', binding, self.engine.get_binding_shape(binding))
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        
    def _softmax(self, input, axis):
        #inference functions
        exps = np.exp(input)
        sums = np.expand_dims(np.sum(exps, axis=axis),-1)
        return exps/sums

    def _reshape_outputs(self, outputs):
        outputs_new = np.reshape(outputs, [self.batch_size, -1])
        return outputs_new
        
    def infer(self, raw_image_generator):
        start = time.time()
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()

        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(self.host_inputs[0], batch_input_image.ravel())
        end = time.time()
        preproc_time = end - start
        
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # Run inference.
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        end = time.time()
        exec_time = end - start

        start = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Do postprocess
        probs = self._softmax(self._reshape_outputs(self.host_outputs[0]), axis=1)
        pred_id = np.argmax(probs, axis=1)
        end = time.time()
        postproc_time = end - start
        return {'probs': probs, 'cls_id':pred_id}, {'pre':preproc_time*1000, 'exec':exec_time*1000, 'post':postproc_time*1000}

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, image_raw):
        """
        description: Convert BGR image to RGB,
                     NO resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, c = image_raw.shape
        assert h==self.input_h
        assert w==self.input_w

        image = image_raw.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1]) #[3,224,224]
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-e','--engine_file', required=True, help='the engine file path')
    ap.add_argument('-i','--image_path', required=True, help='the path to the testing images')
    args = vars(ap.parse_args())
    # load engine
    engine_file_path = args['engine_file']
    image_dir = args['image_path']

    if not os.path.isfile(engine_file_path):
        raise Exception('engine file does not exist')
    if not os.path.isdir(image_dir):
        raise Exception('image folder does not exist')

    # a TRT instance
    efficientNet_wrapper = EfficientNetTRT(engine_file_path)
    try:
        print('batch size is', efficientNet_wrapper.batch_size)

        #warm up 10 times
        for i in range(10):
            _, use_time = efficientNet_wrapper.infer(efficientNet_wrapper.get_raw_image_zeros())
            print('warm_up->{}, time->{:.2f}ms'.format([efficientNet_wrapper.batch_size, 3, efficientNet_wrapper.input_h, efficientNet_wrapper.input_w], use_time['exec']))
        
        #do inference
        total = 0
        correct = 0
        proc_times = []
        pre_times = []
        post_times = []
        batches,labels = get_img_path_batches(efficientNet_wrapper.batch_size, image_dir)
        for batch,labels in zip(batches,labels):
            preds, use_time = efficientNet_wrapper.infer(efficientNet_wrapper.get_raw_image(batch))
            pred_ids = preds['cls_id']
            temp = pred_ids == np.array(labels)
            correct += temp.sum()
            total += len(labels)
            
            print('input->{}, exec time->{:.2f}ms'.format(batch, use_time['exec']))
            pre_times.append(use_time['pre'])
            proc_times.append(use_time['exec'])
            post_times.append(use_time['post'])
        pre_times = np.array(pre_times)
        proc_times = np.array(proc_times)
        post_times = np.array(post_times)
        print('[INFO] accuracy: {:.2f}'.format(correct/total))
        print('[INFO] mean proc time: {:.2f}ms'.format(proc_times.mean()))
        print('[INFO] max proc time: {:.2f}ms'.format(proc_times.max()))
        print('[INFO] min proc time: {:.2f}ms'.format(proc_times.min()))
        print('[INFO] mean preproc time: {:.2f}ms, mean postproc time: {:.2f}ms'.format(pre_times.mean(), post_times.mean()))
    finally:
        # destroy the instance
        efficientNet_wrapper.destroy()
