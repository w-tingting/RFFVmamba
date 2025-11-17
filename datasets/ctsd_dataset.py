## -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class CTSD(Dataset):
    base_folder = 'CTSD'

    def __init__(self, root_dir, train=False, transform=None, target_transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'Train' if train else 'Test'
        self.csv_file_name = 'Train_Annotation.csv' if train else 'Test_Annotation.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 7]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(classId)

        return img, target
