from keras.applications import VGG16, ResNet50
from keras.layers import Flatten, Dense, Dropout
from keras import Input
from keras import Model
from keras.optimizers import Adam, SGD
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from src.number_detection.preprocess_dataset import load_data
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the training process
vgg.trainable = True

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(1024, activation="relu")(flatten)
bboxHead = Dense(512, activation="relu")(bboxHead)
bboxHead = Dense(256, activation="relu")(bboxHead)
bboxHead = Dense(128, activation="relu")(bboxHead)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

initial_lr = 1e-4
lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=100000, decay_rate=0.96, staircase=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

opt = Adam(learning_rate=lr_schedule)
model.compile(loss="huber_loss", optimizer=opt)
print(model.summary())



# load in training and test data
src_dataset = 'soccernet_dataset_expanded'
train_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\number_detection\\{src_dataset}\\train\\images'
train_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\number_detection\\{src_dataset}\\train\\labels'
test_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\number_detection\\{src_dataset}\\test\\images'
test_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\number_detection\\{src_dataset}\\test\\labels'

train_data = load_data(train_image_folder, train_label_folder)
test_data = load_data(test_image_folder, test_label_folder)

trainImages = np.array(train_data[0], dtype='float32')
trainTargets = np.array(train_data[1], dtype='float32')
testImages = np.array(test_data[0], dtype='float32')
testTargets = np.array(test_data[1], dtype='float32')

train_gen = DataGenerator(trainImages, trainTargets, 8)
test_gen = DataGenerator(testImages, testTargets, 8)

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=20,
    verbose=1,
    callbacks=early_stopping)

model.save('vgg_weights\\vgg16_num_detection_soccernet_expanded3.h5')


# plot the model training history
# N = 10
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.title("Bounding Box Regression Loss on Training Set")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend(loc="lower left")