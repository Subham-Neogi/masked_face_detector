# RECOGNIZING MASKED FACES USING Single Shot Detectors (SSD) and CNN

Using the pretrained [OpenCV’s deep learning](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html) face detector which is based on the [Single Shot Detector (SSD)](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/#download-the-code) framework with a [ResNet](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33) base network and [CNN](https://www.tensorflow.org/tutorials/images/cnn) + Binary Classifier trained on [RMFD](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) dataset, the program identifies whether people are wearing a mask.

## Usage

```bash
python scripts/face_detection_video.py --prototxt models/deploy.prototxt.txt --model models/res10_300x300_ssd_iter_140000.caffemodel --saved checkpoint/
```

## Demo

![MASK DETECTOR DEMO](results/maskdetector.gif)

## Requirements

[Tensorflow 2](https://www.tensorflow.org/install)

```bash
pip install --upgrade pip

pip install tensorflow
```

[matplotlib](https://matplotlib.org/)

```bash
pip install matplotlib
```

[opencv-python](https://pypi.org/project/opencv-python/)

```bash
pip install opencv-python
```

[imutils](https://pypi.org/project/imutils/)

```bash
pip install imutils
```
