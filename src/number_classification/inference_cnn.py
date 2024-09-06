from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
import numpy as np
import cv2

image = load_img('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\soccernet_dataset_cropped\\test\\images\\0_2_jpg.rf.d9fe551c9f6c8660b02db82698e5dc66.jpg')
image = img_to_array(image)
image = cv2.resize(image, (224, 224))

classifier = load_model('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\cnn_weights\\cnn_num_classifier.h5')

preds = classifier.predict(np.array([image]))
print('Predicted class: ', preds.argmax())