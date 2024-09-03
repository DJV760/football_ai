from src.number_detection.inference_vgg import inference_vgg16
import cv2
import os

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