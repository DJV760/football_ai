from src.number_classification.create_classification_dataset import create_number_classification_dataset
from src.number_detection.inference_vgg import inference_vgg16
import matplotlib.pyplot as plt
import tensorflow as tf

def main():

    with tf.device('/CPU:0'):

    ######################################################## VGG model training ###########################################################

    # src_dataset = 'soccernet_dataset_expanded'
    # model_weights = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\vgg_weights\\vgg16_num_detection_soccernet_expanded3.h5'
    # test_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\images'
    # test_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\labels'

    # patch = inference_vgg16(test_image_folder, test_label_folder, model_weights, 55)
    # plt.imshow(patch)
    # plt.show()

    ############################################### Number classification dataset creation ############################################### 
    
        src_dataset = 'soccernet_dataset_expanded'
        origin_image_dir = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\images'
        origin_label_dir = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\labels'
        weights_vgg = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\vgg_weights\\vgg16_num_detection_soccernet_expanded3.h5'
        destination_dir = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\soccernet_dataset_cropped\\train\\images'

        create_number_classification_dataset(origin_image_dir, origin_label_dir, weights_vgg, destination_dir)



if __name__ == "__main__":
    main()