# image reader
from PIL import Image
import matplotlib.pyplot as plt

## torch module
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

## user custom dataset
from torch.utils.data import Dataset, DataLoader

## file metadata module
import os.path
from os import listdir
from os.path import isfile, join

## file path
TRAIN_FOLDER_PATH = "E:\\map_tile_checker"
## train var
TRAIN_DATA_LABEL = ['line', 'square', 'unspecified_shapes', 'dispersion']
CLASS_COUNT = 4
TRAIN_BATCH_SIZE = 4
TRAIN_EPOCH = 1
TRAIN_LEARINING_LATE = 0.001


class CustomDataset(Dataset):

    def init_image_data(self):
        ## get image file list and append dataset
        image_dataset = {}

        for label_name in self.label_array:
            image_dataset[label_name] = self.get_filelist(os.path.join(self.source_path, label_name))   

        return image_dataset

    def get_filelist(self, file_path):
        ## get file path list
        if os.path.exists(file_path):
            return [
                file_name for file_name in listdir(file_path) if isfile(join(file_path, file_name))
            ]
        else:
            return None

    def convert_data(self):
        all_labels = []
        all_img_file_path = []
        length = 0

        for label_item in self.label_array:
            if(self.data_image_path[label_item] != None):
                for image_path in self.data_image_path[label_item]:
                    all_labels.append(label_item)
                    all_img_file_path.append(
                        os.path.join(self.source_path, label_item, image_path)
                    )
                    length += 1
        return all_labels, all_img_file_path, length
    
    def __init__(self, labels, data_source_path, transforms=None):
        self.label_array = labels
        self.transforms = transforms
        self.source_path = data_source_path
        self.data_image_path = self.init_image_data()
        
        ## 
        self.all_label_array, self.all_image_array, self.length = self.convert_data()
        self.num_classes = len(labels)


    def __getitem__(self, index):
        #return image, label
        #image_path = self.data_image_path[label_name][index]
        #img = Image.open(image_path)

        img = Image.open(self.all_image_array[index])
        if self.transforms is not None:
            img = self.transforms(img)

        return { 'image' : img, 'label' : self.all_label_array[index] }

    def __len__(self):
        return self.length


class CNN_network(nn.Module):

    def __init__(self, label_array):
        super(CNN_network, self).__init__()
        
        ## network
        self.start_layer = self.conv_module(3, 16)
        self.hidden_layer_array = [
            self.conv_module(16, 32),
            self.conv_module(32, 64),
            self.conv_module(64, 128),
            self.conv_module(128, 256)
        ]
        self.last_layer = self.global_avg_pool(256, len(label_array))
        self.class_num = len(label_array)

    def forward(self, x):
        ##TODO : network forward
        out = self.start_layer(x)
        for layer_item in self.hidden_layer_array :
            out = layer_item(out)
        out = self.last_layer(out)
        out.view(-1, len(self.class_num))

        return out

    def conv_module(self, in_num, out_num):
        ## set conv2d layer
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
    

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )



if __name__ == "__main__":
    #init_transforms

    transforms_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomRotation(10.),
            transforms.ToTensor()
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]
    )

    #init_train_data()
    train_data = CustomDataset(
        TRAIN_DATA_LABEL,
        os.path.join(TRAIN_FOLDER_PATH, 'train_data'),
        transforms=transforms_train
    )
    train_data_loader = DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)
    
    '''
    train_data_dump = CustomDataset(
        TRAIN_DATA_LABEL,
        os.path.join(TRAIN_FOLDER_PATH, 'train_data_dump'),
        transforms=transforms_train
    )
    train_data_dump_loader = DataLoader(train_data_dump, batch_size = TRAIN_BATCH_SIZE, shuffle = True)
    '''

    test_data = CustomDataset(
        TRAIN_DATA_LABEL,
        os.path.join(TRAIN_FOLDER_PATH, 'test_data'),
        transforms=transforms_test
    )
    test_data_loader = DataLoader(test_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)


    ## device, network init
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network_model = CNN_network(TRAIN_DATA_LABEL).to(device)

    ## update var
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network_model.parameters(), lr=TRAIN_LEARINING_LATE)

    for epoch_count in range(TRAIN_EPOCH):
        for batch_size, data_set in enumerate(train_data_loader):
            images = data_set['image'].to(device)
            labels = data_set['label'].to(device)

            ## network pass
            outputs = network_model(images)
            loss = criterion(outputs, labels)

            ## update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_size + 1) % TRAIN_BATCH_SIZE == 0:
                print(f'Epoch {epoch_count} / {TRAIN_BATCH_SIZE}, Loss : {loss.item():.4f}')


