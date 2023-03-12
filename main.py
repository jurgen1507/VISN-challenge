import matplotlib.pyplot as plt
import matplotlib.patches as pat
from sklearn import datasets
from sklearn import svm
import tensorflow as tf
import random
import numpy as np

class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd', 'Qh', 'Qs']

#convert bounding box to min, max of x and y
def convert_to_min_max(bbox2, width, height):
    bbox1 = bbox2.split()
    bbox = [float(i) for i in bbox1]
    print(bbox)
    xmin, ymin = bbox[1]-bbox[3]/2, bbox[2]-bbox[4]/2
    xmax, ymax = bbox[1]+bbox[3]/2, bbox[2]+bbox[4]/2
    return [xmin*width, ymin*height, xmax*width, ymax*height]

def plot_bounding_box(imagedir, bbox, labels=[]):
    image= plt.imread(imagedir)
    bounding_boxes = []
    with open(bbox, 'r') as f:
        bounding_boxes.append(f.readlines())
    height, width, c = image.shape
    bboxco = []
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    patches = []
    for n, i in enumerate(bounding_boxes[0]):
        bboxco.append(convert_to_min_max(i, width, height))
        # colors.append([random.random(), random.random(), random.random()])
        # x1pos = int(x1*width)
        # x2pos = int(x2*width)
        # y1pos = int(y1*height)
        # y2pos = int(y2*height)
        # width = x2pos-x1pos
        # height = y2pos-y1pos
        
        # class_name = class_names[int(labels[n])]
        # pat.Rectangle((x1, y1), width, height)
    for b in bboxco:
        patches.append(pat.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1, edgecolor='r', facecolor='none'))
    print(bboxco)
    return patches
from PIL import Image

result = plot_bounding_box("data/train/images/000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.jpg", "data/train/labels/000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.txt")
plt.imshow(Image.open("data/train/images/000090528_jpg.rf.d50e89610e5c97c61632c290692f3e75.jpg"))
ax = plt.gca()
for r in result:
    ax.add_patch(r)
plt.show()




data = tf.keras.utils.image_dataset_from_directory("data/train/", 
                                                   labels="inferred",
                                                   label_mode="int",
                                                   
                                                   )

