from src.number_detection.preprocess_dataset import load_detection_data, load_classification_data
from src.number_classification.create_classification_dataset import create_number_classification_dataset, create_labels_for_dataset
from src.number_detection.inference_vgg import inference_vgg16
from src.number_classification.model import train_cnn
import matplotlib.pyplot as plt
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def main():

    with tf.device('/GPU:0'):

    ################################################### Number detection dataset creation #################################################

        # src_dataset = 'soccernet_dataset_expanded'
        # origin_image_dir = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\test\\images'
        # origin_label_dir = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\test\\labels'
        # data = load_data(origin_image_dir, origin_label_dir, 'number_detection')

    ######################################################## VGG model training ###########################################################

    # src_dataset = 'soccernet_dataset_expanded'
    # model_weights = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\vgg_weights\\vgg16_num_detection_soccernet_expanded3.h5'
    # test_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\images'
    # test_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\labels'

    # patch = inference_vgg16(test_image_folder, test_label_folder, model_weights, 55)
    # plt.imshow(patch)
    # plt.show()

    ############################################### Number classification dataset creation ############################################### 
    
        # src_dataset = 'soccernet_dataset_expanded'
        # origin_image_dir = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\images'
        # origin_label_dir = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\{src_dataset}\\train\\labels'
        # weights_vgg = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\vgg_weights\\vgg16_num_detection_soccernet_expanded3.h5'
        # destination_dir = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\soccernet_dataset_cropped\\train\\images'

        # create_number_classification_dataset(origin_image_dir, origin_label_dir, weights_vgg, destination_dir)

        # origin_image_dir = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\soccernet_dataset_expanded\\train\\images'
        # origin_label_dir = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\soccernet_dataset_expanded\\train\\labels'
        # destination_image_dir = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\soccernet_dataset_cropped\\train\\images'
        # destination_label_dir = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\soccernet_dataset_cropped\\train\\labels'
        # yaml_path = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_detection\\soccernet_dataset\\data.yaml'

        # create_labels_for_dataset(origin_label_dir, destination_label_dir, origin_image_dir, destination_image_dir, yaml_path)
    
    ########################################################## CNN training #############################################################

    # load in training and test data
        src_dataset = 'soccernet_dataset_cropped'
        train_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\{src_dataset}\\train\\images'
        train_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\{src_dataset}\\train\\labels'
        test_image_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\{src_dataset}\\test\\images'
        test_label_folder = f'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\src\\number_classification\\{src_dataset}\\test\\labels'

        train_cnn(train_image_folder, train_label_folder, test_image_folder, test_label_folder)

if __name__ == "__main__":
    main()