from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.utils import Sequence
from src.number_detection.preprocess_dataset import load_classification_data
import tensorflow as tf
import numpy as np

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
    
def train_cnn(train_image_folder, train_label_folder, test_image_folder, test_label_folder):

    train_data = load_classification_data(train_image_folder, train_label_folder, 'train')
    test_data = load_classification_data(test_image_folder, test_label_folder, 'test')

    trainImages = np.array(train_data[0], dtype='int')
    trainTargets = np.array(train_data[1], dtype='int')
    testImages = np.array(test_data[0], dtype='int')
    testTargets = np.array(test_data[1], dtype='int')

    batch_size = 8
    train_gen = DataGenerator(trainImages, trainTargets, batch_size)
    test_gen = DataGenerator(testImages, testTargets, batch_size)

    #CNN architecture

    # Initialising the CNN
    classifier = Sequential()

    classifier.add(Conv2D(128, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Flatten())

    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 64, activation = 'relu'))
    classifier.add(Dense(units = 64, activation = 'relu'))
    classifier.add(Dense(units = len(trainTargets[0]), activation = 'softmax'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    #Data Augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=30,
        shear_range=0.5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(trainImages)

    epochs = 80

    H = classifier.fit(
        datagen.flow(trainImages, trainTargets, batch_size=batch_size),
        steps_per_epoch=len(train_gen),
        validation_data=test_gen,
        validation_steps=len(test_gen),
        epochs=epochs,
        verbose=2
    )

    classifier.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\cnn_weights\\cnn_num_classifier.h5')