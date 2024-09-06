from src.number_detection.inference_vgg import inference_vgg16
import cv2
import os
import yaml

def create_number_classification_dataset(origin_image_dir: str, origin_label_dir: str, weights: str, destination_dir: str) -> None:

    '''
    This function creates dataset that will be used for number classification model training. It is created utilizing trained vgg bbox prediction model.
    The patch of original image that corresponds to the predicted bbox would be extracted and it will represent a sample of newly created dataset. 
    This process is repeated for all of the images in number detection dataset

    Parameters:
        origin_image_dir (str): Path to the images of number detection dataset
        origin_label_dir (str): Path to the labels of number detection dataset
        weights (str): Path to the weights obtained during vgg model training (now used for inference)
        destination_dir (str): Path to the images of number classification dataset that are yet to be created

    Returns:
        None
    '''

    image_files = [f for f in os.listdir(origin_image_dir) if os.path.isfile(os.path.join(origin_image_dir, f))]

    try:
        for i, image in enumerate(image_files):
            patch = inference_vgg16(origin_image_dir, origin_label_dir, weights, i, 'dataset_creation')
            cv2.imwrite(destination_dir + '\\' + image, patch)
    except:
        print('Dataset not created successfully!')

def create_labels_for_dataset(origin_label_dir: str, destination_label_dir: str, origin_image_dir: str, destination_image_dir: str, yaml_path: str) -> None:

    '''
    Function that creates appropriate labels for new dataset, according to labels from original dataset. It is important to mention that original 
    soccernet labels are mapped to label values from jersey_number_detection_dataset_3, by reading their mapped values from .yaml metadata file. 
    That way label class inconsistency is being solved.

    Parameters:
        origin_label_dir (str): Path to the labels of detection dataset
        destination_label_dir (str): Path to the labels of classification dataset
        origin_image_dir (str): Path to the images of detection dataset
        destination_image_dir (str): Path to the images of classification dataset
        yaml_path (str): Path to the metadata file which contains ground truth classes of labels

    Returns:
        None
    '''
    
    label_files_destination = []
    label_files_origin = [f for f in os.listdir(origin_label_dir) if os.path.isfile(os.path.join(origin_label_dir, f))]
    image_files_origin = [f for f in os.listdir(origin_image_dir) if os.path.isfile(os.path.join(origin_image_dir, f))]
    image_files_destination = [f for f in os.listdir(destination_image_dir) if os.path.isfile(os.path.join(destination_image_dir, f))]

    for image in image_files_origin:
        if image in image_files_destination:
            corresponding_label = image.replace('.jpg', '.txt')
            if corresponding_label in label_files_origin:
                label_origin_index = label_files_origin.index(corresponding_label)
                label_files_destination.append(label_files_origin[label_origin_index])

    for label in label_files_destination:
        number_class = process_label_for_classification(os.path.join(origin_label_dir, label), yaml_path)
        with open(os.path.join(destination_label_dir, label), 'w') as file:
            file.write(number_class)

def process_label_for_classification(label_path: str, yaml_path: str) -> str:

    '''
    Helper function that processes labels if needed, so that they are truthful afterwards

    Parameters:
        label_path (str): Path to the label that will be processed
        yaml_path (str): Path to the metadata file which contains ground truth classes of labels

    Returns:
        number_class (str): Ground truth class of a label, that will be written in the corresponding .txt file and used during classification 
        model training
    '''

    with open(label_path, 'r') as file:
        number_class = file.read().split(' ')[0]
        if label_path.split('\\')[-1][0].isdigit():
            classes_metadata = read_yaml_file(yaml_path)
            number_class = map_false_class(classes_metadata, number_class)
        return number_class

def read_yaml_file(yaml_path: str) -> list:

    '''
    Helper function that gets mapped classes from .yaml file

    Parameters:
        yaml_path (str): Path to the metadata file which contains ground truth classes of labels

    Returns:
        list: List of mapped classes
    '''

    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file).get('names', [])
    
def map_false_class(classes: list, number_class: str) -> str:

    '''
    Helper function that maps classes to ground truth values

    Parameters:
        classes (list): List of mapped classes
        number_class (str): Value that is extracted from original label files

    Returns:
        str: Ground truth number of a class
    '''

    return classes[int(number_class)]