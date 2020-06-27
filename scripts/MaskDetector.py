import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import numpy as np
import argparse

class MaskDetector:

    def __init__(self):
        self.IMG_HEIGHT = 32
        self.IMG_WIDTH = 32
        self.checkpoint_path=None
        self.image_path=None
        self.img=None
        self.model=None

    def create_model(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))


        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model.load_weights(self.checkpoint_path)
        self.model = model

    def decode_img_from_file(self, image_path):
        self.image_path = image_path
        img = tf.io.read_file(self.image_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])
        self.img = tf.reshape(img, [1, 32, 32,3])
    
    def decode_img(self, img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])
        self.img = tf.reshape(img, [1, 32, 32,3])
    
    def predict(self):
        return self.model.predict(self.img) 

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-c", "--checkpoint", required=True, help="path where checkpoint file is stored")
    args = vars(ap.parse_args())  

    mask_detector = MaskDetector()
    mask_detector.create_model(args["checkpoint"])
    mask_detector.decode_img_from_file(args["image"])
    #print(decode_img(test_image).shape)
    predicted = mask_detector.predict()
    print(predicted)
    print(np.argmax(predicted[0]))