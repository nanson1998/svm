import os
import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


config = {
    # e.g. train/val/test set should be located in os.path.join(config['datapath'], 'train/val/test')
    "datapath": "data"
}


class CustomDataset(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase="train", shape=(256, 256)):
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.data_path = os.path.join(config["datapath"], phase)
        self.data_list = self.create_data_list(self.data_path)

        self.shape = shape
        self.config = config

        # transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(self.shape[0], self.shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        item = self.data_list[idx]
        label = item[1]

        image = Image.open(item[0]).convert("RGB")  # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]

        return image, label

    def __len__(self):
        return len(self.data_list)

    def create_data_list(self, data_path):
        ret = []

        classes = glob.glob(data_path + "/*")
        for i, cl in enumerate(classes):
            imgs = glob.glob(cl + "/*")
            temp = [(path, i) for path in imgs]
            ret.extend(temp)

        return ret
