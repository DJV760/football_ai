import numpy as np
import os
import cv2
from typing import Optional


def load_data(image_dir, label_dir) -> list:

    ''' This function serves as data loader. For the given image and label dirs, it returns data 
    in form of a vector of 3 more vectors, each representing images, labels and filenames respectively'''

    image_list = []
    label_list = []
    filename_list = []

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        label = load_label(label_path)

        if label is not None:
            image_list.append(image)
            label_list.append(label)
            filename_list.append(filename)

            data = [image_list, label_list, filename_list]

    return data

def load_label(label_path) -> Optional[np.ndarray]:

    ''' Helper function which extracts and formats labels to numpy arrays '''

    with open(label_path, 'r') as file:
        content = file.read().split(' ')
        cls_id_label = content[0]
        bbox_coordinates = content[-4:]

        try:
            return np.array(bbox_coordinates, dtype='float32')
        except:
            return None

# train_image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\soccernet_dataset\\train\\images'
# train_label_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\soccernet_dataset\\train\\labels'
# test_image_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\soccernet_dataset\\test\\images'
# test_label_folder = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\soccernet_dataset\\test\\labels'

# train_data = load_data(train_image_folder, train_label_folder)
# test_data = load_data(test_image_folder, test_label_folder)
# print(train_data[2][2])
# print(train_data[1][2])
# plt.imshow(train_data[0][2])
# plt.show()