## file metadata module
import os.path
import shutil
from os import listdir
from os.path import isfile, join


BYNARY_FILE_PATH = 'E:\\map_tile_checker\\train_data'
TEST_FILE_PATH = 'E:\\map_tile_checker\\test_data'
LABEL_ARRAY = ['dispersion', 'line', 'square', 'unspecified_shapes']
TEST_FOLDER_CONTEXT = 'alpha_distribution_'

def init_image_data(source_path):
    ## get image file list and append dataset
    image_dataset = {}

    for label_name in LABEL_ARRAY:
        image_dataset.update(get_filelist(os.path.join(source_path, label_name), label_name)) 

    return image_dataset

def get_filelist(file_path, label):
    ## get file path list
    result_dic = {}
    if os.path.exists(file_path):
        for file_name in listdir(file_path):
            if isfile(join(file_path, file_name)):
                result_dic[file_name] = label

    return result_dic

def get_testfile_list(source_path, context, test_dict):
    folder_array = [
        f'{context}{folder_name}' for folder_name in range(0, 16) 
    ]

    for folder_name in folder_array:
        file_path = os.path.join(source_path, folder_name)
        for file_name in listdir(file_path):
            if isfile(join(file_path, file_name)):
                vender = test_dict[file_name]
                shutil.copy(f'{file_path}\\{file_name}', f'{source_path}\\{vender}')

        print(f'work end! {folder_name}')
    

if __name__ == "__main__":
    image_file_path_dict = init_image_data(BYNARY_FILE_PATH)
    get_testfile_list(TEST_FILE_PATH, TEST_FOLDER_CONTEXT, image_file_path_dict)


