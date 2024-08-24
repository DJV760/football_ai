from preprocess_dataset import load_data
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
from keras.src.utils import img_to_array
from keras.src.saving import load_model

test_image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\test\\images'
test_label_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\test\\labels'

test_data = load_data(test_image_folder, test_label_folder)

testImages = np.array(test_data[0])
testTargets = np.array(test_data[1])
testFilenames = test_data[2]

image = testImages[0]
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

model = load_model(filepath='C:\\Users\\z0224841\\PycharmProjects\\football_ai\\vgg16_num_detection.h5')

preds = model.predict(image)
preds_bbox = preds[0]

(startX, startY, endX, endY) = preds_bbox

# load the input image (in OpenCV format), resize it such that it
# fits on our screen, and grab its dimensions
image = cv2.imread('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset\\test\\images\\' + testFilenames[2])
image = imutils.resize(image)
(h, w) = image.shape[:2]


# scale the predicted bounding box coordinates based on the image
# dimensions
startX = int(startX * w)
startY = int(startY * h)
endX = int(endX * w)
endY = int(endY * h)

# draw the predicted bounding box on the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
plt.imshow(image)
plt.show()