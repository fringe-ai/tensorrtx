"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

CONF_THRESH = 0.7
MAX_NUM_BBOXES = 10
MASK_EDGE_LENGTH = 14


def get_img_path_batches(batch_size, img_dir, fmt='png'):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if os.path.splitext(name)[-1] != '.'+fmt:
                continue
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def plot_one_box_and_mask(box, mask, img, color=None, label=None, line_thickness=None):
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
    x1,y1,x2,y2 = box.astype("int")
    c1, c2 = (x1, y1), (x2, y2)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    blended = ((0.4 * np.array(color,dtype=np.float)) + (0.6 * img[y1:y2,x1:x2,:][mask])).astype("uint8")
    img[y1:y2,x1:x2,:][mask] = blended
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
        


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = {}
        cuda_outputs = {}
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_c = engine.get_binding_shape(binding)[2]
                self.input_w = engine.get_binding_shape(binding)[1]
                self.input_h = engine.get_binding_shape(binding)[0]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs[binding] = host_mem
                cuda_outputs[binding] = cuda_mem

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, self.input_h, self.input_w, self.input_c])
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for k in cuda_outputs:
            cuda.memcpy_dtoh_async(host_outputs[k], cuda_outputs[k], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        
        # Do postprocess
        for i in range(self.batch_size):
            result_scores = np.array(host_outputs['scores'][i*MAX_NUM_BBOXES:(i+1)*MAX_NUM_BBOXES])
            result_classid = np.array(host_outputs['labels'][i*MAX_NUM_BBOXES:(i+1)*MAX_NUM_BBOXES])
            result_boxes = host_outputs['boxes'][i*MAX_NUM_BBOXES*4:(i+1)*MAX_NUM_BBOXES*4]
            result_boxes = np.reshape(result_boxes, (MAX_NUM_BBOXES,4))
            result_masks = host_outputs['masks'][i*MAX_NUM_BBOXES*MASK_EDGE_LENGTH**2:(i+1)*MAX_NUM_BBOXES*MASK_EDGE_LENGTH**2]
            result_masks = np.reshape(result_masks, (MAX_NUM_BBOXES,MASK_EDGE_LENGTH,MASK_EDGE_LENGTH))

            # filter those less than confidence threshold
            scores_mask = result_scores>CONF_THRESH
            result_scores = result_scores[scores_mask]
            result_classid = result_classid[scores_mask]
            result_boxes = result_boxes[scores_mask]
            result_masks = result_masks[scores_mask]

            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                # resize the masks
                box,mask = result_boxes[j], result_masks[j]
                x1,y1,x2,y2 = box.astype(np.int)
                w,h = x2-x1,y2-y1
                mask = cv2.resize(mask,dsize=(w,h))
                plot_one_box_and_mask(
                    box, mask>CONF_THRESH,
                    batch_image_raw[i],
                    label="{}:{:.2f}".format(
                        int(result_classid[j]), result_scores[j]
                    ),
                )
        return batch_image_raw, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        
        image = image.astype(np.float32)
        # # Normalize to [0,1]
        # image /= 255.0
        # # HWC to CHW format:
        # image = np.transpose(image, [2, 0, 1])
        # # CHW to NCHW format
        # image = np.expand_dims(image, axis=0)
        # # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w



class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('outputs', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))



if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "mask.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    

    if os.path.exists('outputs/'):
        shutil.rmtree('outputs/')
    os.makedirs('outputs/')
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    try:
        print('batch size is', yolov5_wrapper.batch_size)
        
        image_dir = "data/test"
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(yolov5_wrapper, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()