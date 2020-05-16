## Masked RCNN
#  https://github.com/matterport/Mask_RCNN
# tutorial: https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
from matplotlib import pyplot
from matplotlib.patches import Rectangle
%matplotlib inline

#define our own config class by overloading the Mask RCNN config class
class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_inference"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class
    # names)
    NUM_CLASSES = 1+80

config = myMaskRCNNConfig()

print("loading  weights for Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

#load the weights trained on coco dataset
model.load_weights('mask_rcnn_coco.h5', by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',\
               'bus', 'train', 'truck', 'boat', 'traffic light',\
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',\
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']


# drawing images with bounding boxes
def draw_image_with_boxes(filename, boxes_list):
    # load the image
    data = pyplot.imread(filename)
    
    # plot the image
    pyplot.imshow(data)

    # get the context for drawing boxes
    ax = pyplot.gca()
    
    # plot each box
    for box in boxes_list:
        # get coordinates
        y1, x1, y2, x2 = box
        
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)

        # draw the box
        ax.add_patch(rect)

        # show the plot
        pyplot.show()


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

img_path = r'C:\Users\schoud790\Documents\Python Codes\computer vision\datasets\traffic data\Traffic - 20581.mp4'
vid_cap = cv2.VideoCapture(img_path)

success, frame = vid_cap.read()

cv2.imwrite("image.jpg", frame)

img = load_img('image.jpg')

pyplot.imshow(img)
img = img_to_array(img)

# make prediction
results = model.detect([img], verbose=0)

# visualize the results
draw_image_with_boxes('image.jpg', results[0]['rois'])

## draw the mask
# get dictionary for first prediction
from mrcnn.visualize import display_instances
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


classes= r['class_ids']
print("Total Objects found", len(classes))
for i in range(len(classes)):
    print(class_names[classes[i]])

vid_cap.release()