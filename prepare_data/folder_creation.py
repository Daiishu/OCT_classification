import re
import os
import shutil


def create_folder_for_data(name='dataset', path='../../') -> str:
    """
    Function to create folder for new dataset

    :param name: Folder name, default dataset
    :param path: Path of folder, default '../../'
    :return: name of folder
    """
    if not os.path.exists(path + name):
        os.mkdir(path + name)
    else:
        if len(re.search(r"\d*$", name).group(0)) != 0:
            num = int(re.search(r"\d*$", name).group(0))
            num += 1
            name = re.search(r"\D*", name).group(0) + str(num)
        else:
            name = name + "_1"
        if not os.path.exists(path + name):
            os.mkdir(path + name)
        else:
            return create_folder_for_data(name)
    return name


def remove_all_folders_with_name(name='dataset', path='../../', without='') -> None:
    """
    Function to remove all folders created with function: create_folder_for_data

    :param name: Source name of folders, default dataset
    :param path: Path of folders, default '../../'
    :param without: Folder to not remove
    :return: None
    """
    all_folders = os.listdir(path)
    to_remove = [i for i in all_folders if name in i and i != without]
    for i in to_remove:
        shutil.rmtree(path + i, ignore_errors=True)
