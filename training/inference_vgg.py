from preprocess_dataset import load_data
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
from keras.src.utils import img_to_array
from keras.src.saving import load_model

src_dataset = 'soccernet_dataset'
test_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\{src_dataset}\\test\\images'
test_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\{src_dataset}\\test\\labels'

test_data = load_data(test_image_folder, test_label_folder)

testImages = np.array(test_data[0])
testTargets = np.array(test_data[1])
testFilenames = test_data[2]

image = testImages[0]
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(filepath='C:\\Users\\z0224841\\PycharmProjects\\football_ai\\vgg16_num_detection_soccernet_expanded2.h5')

preds = model.predict(image)
preds_bbox = preds[0]

(startXbb, startYbb, endXbb, endYbb) = preds_bbox

# load the input image (in OpenCV format), resize it such that it fits on our screen, and grab its dimensions
test_image_path = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\{src_dataset}\\test\\images\\' + testFilenames[45]
image = cv2.imread(test_image_path)
image = imutils.resize(image)
(h, w) = image.shape[:2]


# scale the predicted bounding box coordinates based on the image
# dimensions
startX = int((startXbb - endXbb/2) * w)
startY = int((startYbb + endYbb/2) * h)
endX = int((startXbb + endXbb/2) * w)
endY = int((startYbb - endYbb/2) * h)

# draw the predicted bounding box on the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)

# show the output image
plt.imshow(image)
plt.show()