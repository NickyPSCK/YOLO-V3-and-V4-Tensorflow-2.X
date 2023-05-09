# YOLO V3 and V4

## Overview
The YOLO sourcecode in repository is refactored version of this repository: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 by make it more "ease of Use".

## Example and Tutorial
This example is a part of "YOLO_02_tutorial_custom_model_training.ipynb"
for more information please run thru all tutorial notebooks

Load all required packages
```python
import cv2
import matplotlib.pyplot as plt
from yolo.tools import XMLtoInputConvertor
from yolo.data import YOLODatasetGenerator
from yolo.estimator import YOLOEstimator
from yolo.detector import YOLOObjectDetector
```

Convert XML anotations to YOLO input format

```python
IC = XMLtoInputConvertor()

IC.convert(XML_dir='example_trainset/apple.v1-apple-dataset.voc/train',
           annotation_path='example_trainset/apple.v1-apple-dataset.voc/processed_anotation/train.txt')

IC.convert(XML_dir='example_trainset/apple.v1-apple-dataset.voc/valid',
           annotation_path='example_trainset/apple.v1-apple-dataset.voc/processed_anotation/valid.txt')

IC.export_class_names(class_names_path='example_trainset/apple.v1-apple-dataset.voc/processed_anotation/class.txt')
```

Read data

```python
version = 'v4_tiny'

train_data = YOLODatasetGenerator(anotation_path='example_trainset/apple.v1-apple-dataset.voc/processed_anotation/train.txt',
                                  version=version,
                                  input_sizes=416,
                                  batch_size=4,
                                  data_aug=False,
                                  load_images_to_ram=True,
                                  anchor_per_scale=3,
                                  max_bbox_per_scale=100)

valid_data = YOLODatasetGenerator(anotation_path='example_trainset/apple.v1-apple-dataset.voc/processed_anotation/valid.txt',
                                  version=version,
                                  input_sizes=416,
                                  batch_size=3,
                                  data_aug=False,
                                  load_images_to_ram=True,
                                  anchor_per_scale=3,
                                  max_bbox_per_scale=100)
```

Create Model

```python
yolo_custom_estimator = YOLOEstimator(version=version,
                                      input_size=416,
                                      channels=3,
                                      inference_using_pretrained=False,
                                      init_pretrained_weight=True,
                                      disable_gpu=False,
                                      pretrained_weight_dir='./pretrained_weight/')

```

Train Model

```python
yolo_custom_estimator.fit(train_data=train_data,
                          val_data=valid_data,
                          epocs=20,
                          warmup_epocs=5,
                          save_checkpoint=False,
                          save_best_only=True)
yolo_custom_estimator.save(model_dir='custom_model')
```

Make Inference

```python
yolo_custom_detector = YOLOObjectDetector(model=yolo_custom_estimator,
                                          classes=['apple'])

image_path = 'example_trainset/apple.v1-apple-dataset.voc/valid/AnyConv-com__38_jpg.rf.efc76c3d7c82233abb99ad2c7bcc2f05.jpg'
image_rgb = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

result, bboxes = yolo_custom_detector.detect_image(image_rgb=image_rgb,
                                                   score_threshold=0.5,
                                                   iou_threshold=0.5,
                                                   show_label=True,
                                                   show_confidence=True)
plt.imshow(result)
```

## Requirements and Installation

- Install all dependencies
```
pip install -r ./requirements.txt
```

- Download model weight files below into ./pretrained_weight directory

1. yolov3-tiny.weights >> from https://pjreddie.com/media/files/yolov3-tiny.weights
2. yolov3.weights >> from https://pjreddie.com/media/files/yolov3.weights
3. yolov4-tiny.weights >> from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
4. yolov4.weights >> from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


## Interesting Pre-Read/Watch
### Explanation Articles
- https://pylessons.com/YOLOv3-TF2-custrom-train
- https://pylessons.com/YOLOv4-TF2-TensorRT
- https://medium.com/analytics-vidhya/you-only-look-once-yolo-implementing-yolo-in-less-than-30-lines-of-python-code-97fb9835bfd2
- https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

### Explanation VDOs
- https://www.youtube.com/watch?v=c-MVqtfHyAU
- https://www.youtube.com/watch?v=P-P2dwj4zIU
- https://www.youtube.com/watch?v=p44G9_xCM4I

### How to create your own XML anotation with "labelImg"
- https://www.v7labs.com/blog/labelimg-guide
### Example Dataset
- https://universe.roboflow.com/intern/apple-lyngf/dataset/1