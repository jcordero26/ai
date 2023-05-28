

import pathlib
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup train and testing paths
image_path = pathlib.Path("/home/jcordero/work/AI/Dataset")
train_dir = image_path / "train"
test_dir = image_path / "test"

def get_train_dir():
    return train_dir

def get_test_dir():
    return test_dir

def showImg():
    # Set seed
    random.seed(42) # <- try changing this and see what happens

    # 1. Get all image paths (* means "any combination")
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")

    # Turn the image into an array
    img_as_array = np.asarray(img)

    # Plot the image with matplotlib
    plt.figure(figsize=(10, 7))
    plt.imshow(img_as_array)
    plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    plt.axis(False);
    #plt.show(block=True)


# ******************************************** Data Transformation ***************************************************

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(128, 128)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

def explore_dataset():
    # Use ImageFolder to create dataset(s)

    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

    class_names = train_data.classes
    print(class_names)
    class_dict = train_data.class_to_idx
    print(class_dict)


    # Turn train and test Datasets into DataLoaders

    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=1, # how many samples per batch?
                                  num_workers=16, # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                 batch_size=1, 
                                 num_workers=16, 
                                 shuffle=False) # don't usually need to shuffle testing data

    img, label = next(iter(train_dataloader))

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

def get_origa_train_data():
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)
    return train_data


def get_origa_test_data():
    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=data_transform)
    return test_data
