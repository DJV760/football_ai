import os
import shutil

def copy_to_soccernet(dataset_source, dataset_destination):

    ''' This function would get additional samples of data from jersey_number_detection_dataset_3 and paste them to soccernet_dataset '''

    for filename in os.listdir(dataset_source + '\\labels'):

        image_source = dataset_source + f"\\images\\{filename.replace('.txt', '.jpg')}"
        label_source = dataset_source + f"\\labels\\{filename}"
        image_destination = dataset_destination + f"\\images\\{filename.replace('.txt', '.jpg')}"
        label_destination = dataset_destination + f"\\labels\\{filename}"

        with open(label_source, 'r') as file:
            content = file.read().split(' ')
            if len(content) == 5:
                shutil.copy(image_source, image_destination)
                shutil.copy(label_source, label_destination)


dataset_source = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\jersey_number_detection_dataset_3\\train'
dataset_destination = 'C:\\Users\\z0224841\\PycharmProjects\\football_ai\\training\\soccernet_dataset_expanded\\train'
copy_to_soccernet(dataset_source, dataset_destination)