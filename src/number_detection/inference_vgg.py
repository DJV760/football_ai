from src.number_detection.preprocess_dataset import load_detection_data
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils
from keras.utils import img_to_array
from keras.models import load_model

def inference_vgg16(image_folder: str, label_folder: str, weights: str, seed: int, origin='plain_utilization') -> np.ndarray:

    ''' This function performs inference utilizing the trained VGG model '''

    test_data = load_detection_data(image_folder, label_folder)

    testImages = np.array(test_data[0])
    testTargets = np.array(test_data[1])
    testFilenames = test_data[2]

    image = testImages[0]
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    model = load_model(filepath=weights)

    preds = model.predict(image)
    preds_bbox = preds[0]

    (startXbb, startYbb, endXbb, endYbb) = preds_bbox

    # load the input image (in OpenCV format), resize it such that it fits on our screen, and grab its dimensions
    test_image_path = image_folder + '\\' + testFilenames[seed]
    image = cv2.imread(test_image_path)
    image = imutils.resize(image)
    (h, w) = image.shape[:2]

    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int((startXbb - endXbb * 1.2) * w) if origin == 'dataset_creation' else int((startXbb - endXbb/2) * w)
    startY = int((startYbb + endYbb * 1.2) * h) if origin == 'dataset_creation' else int((startYbb + endYbb/2) * h)
    endX = int((startXbb + endXbb * 1.2) * w) if origin == 'dataset_creation' else int((startXbb + endXbb/2) * w)
    endY = int((startYbb - endYbb * 1.2) * h) if origin == 'dataset_creation' else int((startYbb - endYbb/2) * h)

    patch = image[endY+1:startY, startX+1:endX]

    # draw the predicted bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)

    # show the output image
    # plt.imshow(image)
    # plt.show()

    return patch