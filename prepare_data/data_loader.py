import os
import random
import shutil
import prepare_data.folder_creation as folder_creation
from tensorflow import keras


def create_equal_sample_of_data(path_to_original_dataset="../../OCT2017",
                                path='../../', name='dataset') -> str:
    """
    Function to create equal dataset from original data by randomly trimming the data

    :param path_to_original_dataset: path to original data set
    :param path: path to new dataset destination
    :param name: name of creating folder
    :return: path of created data
    """
    path = folder_creation.create_folder_for_data(name=name, path=path)

    shutil.copytree(src=path_to_original_dataset + '/val', dst=path + '/val')
    shutil.copytree(src=path_to_original_dataset + '/test', dst=path + '/test')

    remove_ds_store_files(path=path)

    os.mkdir(path + "/train")

    classes = os.listdir(path_to_original_dataset + "/train")
    class_size = {}

    for cl in classes:
        class_size[cl] = len(os.listdir(path_to_original_dataset + "/train/" + cl))

    min_class_size = min(class_size, key=class_size.get)
    shutil.copytree(src=path_to_original_dataset + '/train/' + min_class_size,
                    dst=path + '/train/' + min_class_size)

    if os.path.exists(path + '/train/' + min_class_size + '/.DS_Store'):
        os.remove(path + '/train/' + min_class_size + '/.DS_Store')
        class_size[min_class_size] -= 1

    for cl in classes:
        if cl != min_class_size:
            os.mkdir(path + "/train/" + cl)
            list_of_images = os.listdir(path_to_original_dataset + "/train/" + cl)
            if '.DS_Store' in list_of_images:
                list_of_images.remove('.DS_Store')
            random_images = random.sample(list_of_images, class_size[min_class_size])
            for image in random_images:
                shutil.copyfile(path_to_original_dataset + "/train/" + cl + "/" + image,
                                path + "/train/" + cl + "/" + image)
    return path


def remove_ds_store_files(path='../../dataset') -> None:
    """
    Function to remove necessary file in dataset

    :param path: path of dataset, default '../../dataset'
    :return: None
    """
    for p in os.listdir(path):
        if os.path.exists(path + '/' + p + '/.DS_Store'):
            os.remove(path + '/' + p + '/.DS_Store')
        for pp in os.listdir(path + '/' + p):
            if os.path.exists(path + '/' + p + '/' + pp + '/.DS_Store'):
                os.remove(path + '/' + p + '/' + pp + '/.DS_Store')


def load_data_using_keras(path='../../dataset', path_to_original_dataset="../../OCT2017") -> tuple:
    """
    Creating tensorflow datasets

    :param path_to_original_dataset: path to original data set
    :param path: path to new dataset destination
    :return: train_ds, val_ds, test_ds
    """

    path = create_equal_sample_of_data(path_to_original_dataset=path_to_original_dataset, path=path)
    train_ds = keras.utils.image_dataset_from_directory(
        directory=path + '/train/',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=32,
        image_size=(256, 256))

    val_ds = keras.utils.image_dataset_from_directory(
        directory=path + '/val/',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=32,
        image_size=(256, 256))

    test_ds = keras.utils.image_dataset_from_directory(
        directory=path + '/test/',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=32,
        image_size=(256, 256))

    return train_ds, val_ds, test_ds
