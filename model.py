import tensorflow as tf
from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.layers import Flatten, Dense, Dropout
from keras import Input
from keras import Model
from keras._tf_keras.keras.optimizers import Adam
from training.preprocess_dataset import load_data
import numpy as np

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = True

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)


opt = Adam(learning_rate=1e-4)
model.compile(loss="mean_squared_error", optimizer=opt)
print(model.summary())

# load in training and test data
train_image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\train\\images'
train_label_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\train\\labels'
test_image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\test\\images'
test_label_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\test\\labels'

train_data = load_data(train_image_folder, train_label_folder)
test_data = load_data(test_image_folder, test_label_folder)

trainImages = np.array(train_data[0])
trainTargets = np.array(train_data[1])
testImages = np.array(test_data[0])
testTargets = np.array(test_data[1])

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=32,
    epochs=25,
    verbose=1)

model.save('vgg16_num_detection.h5')