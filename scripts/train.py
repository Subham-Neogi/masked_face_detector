import tensorflow as tf
from tensorflow.keras import datasets, layers, models

AUTOTUNE = tf.data.experimental.AUTOTUNE

from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=True, help="path to image data set directory for training")
ap.add_argument("-v", "--validation", required=True, help="path to image data set directory for validation")
ap.add_argument("-c", "--checkpoint", required=True, help="path to store checkpoint files")
args = vars(ap.parse_args())

train_data_dir = pathlib.Path(args["train"])

image_count = len(list(train_data_dir.glob('*/*.*')))
print("[INFO] No. of Training Images:",image_count)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
print("[INFO] CLASS_NAMES:",CLASS_NAMES)

test_data_dir = pathlib.Path(args["validation"])

test_image_count = len(list(test_data_dir.glob('*/*.*')))
print("[INFO] No. of Test Images:",test_image_count)

TEST_CLASS_NAMES = np.array([item.name for item in test_data_dir.glob('*')])
print("[INFO] CLASS_NAMES:",TEST_CLASS_NAMES)

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

''' This is slow
train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))
'''

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n][0]].title())
      plt.axis('off')
  plt.show()

# image_batch, label_batch = next(train_data_gen)
# show_batch(image_batch, label_batch)


list_ds = tf.data.Dataset.list_files(str(train_data_dir/'*/*'))

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  if parts[-2] == 'masked_face':
    return [0, 1]
  
  return [1, 0]

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

#print(labeled_ds)
'''
for image, label in labeled_ds.take(5):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
'''

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat(2)

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_ds = prepare_for_training(labeled_ds)

#print(train_ds)
'''
for image, label in train_ds.take(5):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

image_batch, label_batch = next(iter(train_ds))
show_batch(image_batch.numpy(), label_batch.numpy())
'''

test_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'))
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = prepare_for_training(test_ds)

#print (test_ds)
'''
for image, label in test_ds.take(5):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
'''

#model

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

checkpoint_filepath = args["checkpoint"]
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[model_checkpoint_callback])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
