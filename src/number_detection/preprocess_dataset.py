import numpy as np
import os
import cv2
from typing import Optional, List, Union

def load_detection_data(image_dir: str, label_dir: str) -> list:

    '''
    This function serves as data loader for detection data. For the given image and label dirs, it returns data 
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
            label = load_detection_label(label_path)

            if label is not None:
                image_list.append(image)
                label_list.append(label)
                filename_list.append(filename)

            else:
                print(filename)

    return [image_list, label_list, filename_list]

def load_classification_data(image_dir: str, label_dir: str, origin: str):

    '''
    This function serves as data loader for classification data. For the given image and label dirs, it returns data 
    in form of a vector of 3 more vectors, each representing images, labels and filenames respectively 
    
    Parameters:
        image_dir (str): Path to the input image directory
        label_dir (str): Path to the input label directory
        origin (str): String that tells if the function is called for train or test data creation needs

    Returns:
        data (list): A list that represents vector of loaded data (images, labels and filenames)
    '''

    image_list = []
    label_list = []
    filename_list = []

    classification_train_img_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\train_images.npy'
    classification_train_label_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\train_labels.npy'
    classification_train_filename_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\train_filenames.npy'
    classification_test_img_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\test_images.npy'
    classification_test_label_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\test_labels.npy'
    classification_test_filename_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\test_filenames.npy'
    classes_vector_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\classes_vector.npy'

    if (origin == 'train' and \
        not os.path.exists(classification_train_img_path) and \
        not os.path.exists(classification_train_label_path) and \
        not os.path.exists(classification_train_filename_path) and \
        not os.path.exists(classes_vector_path) or \
        origin == 'test' and \
        not os.path.exists(classification_test_img_path) and \
        not os.path.exists(classification_test_label_path) and \
        not os.path.exists(classification_test_filename_path)):

        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))

            if origin == 'train':
                unique_classes = find_unique_class_values(label_dir)
                np.save(classes_vector_path, unique_classes)
            elif origin == 'test':
                try:
                    unique_classes = np.load(classes_vector_path)
                except:
                    print('Classes from training data are not yet obtained!')
            label = load_classification_label(label_path, unique_classes.tolist())

            if label is not None:
                image_list.append(image)
                label_list.append(label)
                filename_list.append(filename)
            else:
                print(filename)

        if len(image_list) > 0 and len(label_list) > 0 and len(filename_list) > 0:
            if origin == 'train':
                np.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\train_images.npy', image_list)
                np.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\train_labels.npy', label_list)
                np.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\train_filenames.npy', filename_list)

                return [image_list, label_list, filename_list], unique_classes
            
            elif origin == 'test':
                np.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\test_images.npy', image_list)
                np.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\test_labels.npy', label_list)
                np.save('C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\classification_data\\test_filenames.npy', filename_list)

                return [image_list, label_list, filename_list]

    else:

        return load_already_existing_classification_data(classification_train_img_path, classification_train_label_path, classification_train_filename_path, classification_test_img_path, classification_test_label_path, classification_test_filename_path,origin)

def load_already_existing_classification_data(classification_train_img_path: str, classification_train_label_path: str, classification_train_filename_path: str, classification_test_img_path: str, classification_test_label_path: str, classification_test_filename_path: str,origin: str) -> list:

    ''' 
    Helper function which is called in case the load_classification_data function is called over already existing data.

    Parameters:
        classification_train_img_path (str): Path to the training images
        classification_train_label_path (str): Path to the training labels
        classification_train_filename_path (str): Path to the training filenames
        classification_test_img_path (str): Path to the test images
        classification_test_label_path (str): Path to the test labels
        classification_test_filename_path (str): Path to the test filenames
        origin (str): String which tells if the load_classification_data() is called on training or test data
    
    Returns:
        list: List which represents loaded data vector
    '''

    if origin == 'train':
        image_list = np.load(classification_train_img_path) 
        label_list = np.load(classification_train_label_path)
        filename_list = np.load(classification_train_filename_path)
    
    elif origin == 'test':
        image_list = np.load(classification_test_img_path) 
        label_list = np.load(classification_test_label_path)
        filename_list = np.load(classification_test_filename_path)
        
    return [image_list, label_list, filename_list]

def load_detection_label(label_path: str) -> Optional[np.ndarray]:

    '''
    Helper function for loading detection data, which extracts and formats labels to numpy arrays 
    
    Parameters:
        label_path (str): Path to the one particular label in label directory 

    Returns:
        ndarray or None: A numpy array which contains annotated bbox coordinates, or None if none are found
    '''

    try:
        with open(label_path, 'r') as file:
            content = file.read().split(' ')
            if len(content) < 1 or len(content) > 5:
                raise ValueError
            bbox_coordinates = content[-4:]
        
        return np.array(bbox_coordinates, dtype='float32')
    except:
        delete_false_data(label_path)
        return None
    
def load_classification_label(label_path: str, unique_classes: list) -> Union[list, None]:

    try:
        with open(label_path, 'r') as file:
            class_id = file.read()
            one_hot_index = unique_classes.index(int(class_id))
            one_hot_vector = np.zeros(len(unique_classes), dtype='int')
            one_hot_vector[one_hot_index] = 1
            
        return one_hot_vector
    except:
        delete_false_data(label_path)
        return None
    
def delete_false_data(label_path: str) -> None:
    label = label_path
    image_path = (label.split('.txt')[0] + '.jpg').replace('labels', 'images')

    with open(label_path, 'r') as file:
        content = file.read()
        if len(content) < 1 or len(content) > 5:
    
            try:
                os.remove(label)
                os.remove(image_path)
            except:
                return
    
def find_unique_class_values(label_folder: str) -> list:

    classes_list = []
    
    for label in os.listdir(label_folder):
        with open(label_folder + '\\' + label, 'r') as file:
            content = file.read()
            classes_list.append(int(content))
        
    return list(set(classes_list))