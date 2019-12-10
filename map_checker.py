# image reader
from PIL import Image
import matplotlib.pyplot as plt

## torch module
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os

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
TRAIN_BATCH_SIZE = 30
TRAIN_EPOCH = 50
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

    def label_convert(self):
        label_dict = {}
        for index, label_item in enumerate(self.label_array):
            label_dict[label_item] = index

        return label_dict

    def convert_data(self):
        all_labels = []
        all_img_file_path = []
        length = 0

        for label_item in self.label_array:
            if(self.data_image_path[label_item] != None):
                for image_path in self.data_image_path[label_item]:
                    all_labels.append(self.label_dict[label_item])
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
        self.label_dict = self.label_convert()
        self.all_label_array, self.all_image_array, self.length = self.convert_data()
        self.num_classes = len(labels)


    def __getitem__(self, index):
        img = Image.open(self.all_image_array[index])
        img = img.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return { 'image' : img, 'label' : self.all_label_array[index] }

    def __len__(self):
        return self.length


class CNN_network(nn.Module):

    def __init__(self, num_class):
        super(CNN_network, self).__init__()
        
        ## network
        self.start_layer = self.conv_module(3, 16)
        self.layer_2 = self.conv_module(16, 32)
        self.layer_3 = self.conv_module(32, 64)
        self.layer_4 = self.conv_module(64, 128)
        self.layer_5 = self.conv_module(128, 256)
        self.last_layer = self.global_avg_pool(256, num_class)
        self.num_class = num_class

    def forward(self, x):
        ##network forward
        out = self.start_layer(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.last_layer(out)
        out = out.view(-1, self.num_class)

        return out

    def conv_module(self, in_num, out_num):
        ## set conv2d layer
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )


def get_dataset(label, file_path, transform):
    return CustomDataset(
        label, file_path, transforms = transform
    )

def get_model_output(model, item_set, device):
    images = data_set['image'].to(device)
    labels = data_set['label'].to(device)
    
    ## network pass
    return labels, network_model(images)    


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
    train_data = get_dataset(
        TRAIN_DATA_LABEL,
        os.path.join(TRAIN_FOLDER_PATH, 'train_data'),
        transforms_train
    ) 
    train_data_loader = DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)
    
    test_data = get_dataset(
        TRAIN_DATA_LABEL,
        os.path.join(TRAIN_FOLDER_PATH, 'test_data'),
        transforms_test
    )
    test_data_loader = DataLoader(test_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)

    ## device, network init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network_model = CNN_network(num_class = len(TRAIN_DATA_LABEL)).to(device)

    ## update var
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network_model.parameters(), lr=TRAIN_LEARINING_LATE)

    for epoch_count in range(TRAIN_EPOCH):
        for batch_size, data_set in enumerate(train_data_loader):
            labels, outputs = get_model_output(network_model, data_set, device)
            loss = criterion(outputs, labels)  

            ## update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_size + 1) % TRAIN_BATCH_SIZE == 0:
                print(f'[Epoch {epoch_count} / {TRAIN_EPOCH}], Loss : [{loss.item():.4f}]')



    network_model.eval()
    
    correct = 0
    total = 0

    for data_set in test_data_loader:
        labels, outputs = get_model_output(network_model, data_set, device)

        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()

        print(f'Test Accuracy of the model on the {total} test images: {100 * correct / total} %')