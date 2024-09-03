import numpy as np
import os
import cv2
from typing import Optional

def load_data(image_dir: str, label_dir: str) -> list:

    '''
    This function serves as data loader. For the given image and label dirs, it returns data 
    in form of a vector of 3 more vectors, each representing images, labels and filenames respectively 
    
    Parameters:
        image_dir (str): Path to the input image directory
        label_dir (str): Path to the input label directory

    Returns:
        data (list): A list that represents vector of loaded data (images, labels and filenames)
    '''

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
        else:
            print(filename)

    return data

def load_label(label_path) -> Optional[np.ndarray]:

    '''
    Helper function which extracts and formats labels to numpy arrays 
    
    Parameters:
        label_path (str): Path to the one particular label in label directory 

    Returns:
        ndarray or None: A numpy array which contains annotated bbox coordinates, or None if none are found
    '''

    try:
        with open(label_path, 'r') as file:
            content = file.read().split(' ')
            cls_id_label = content[0]
            bbox_coordinates = content[-4:]

        return np.array(bbox_coordinates, dtype='float32')
    except:
        delete_false_data(label_path)
        return None

def delete_false_data(label_path):
    label = label_path
    image_path = (label.split('.txt')[0] + '.jpg').replace('labels', 'images')
    
    try:
        os.remove(label)
        os.remove(image_path)
    except:
        return