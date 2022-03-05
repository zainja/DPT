import numpy as np
import torch.utils.data as data
import os
from PIL import Image


def process_relative_coords_file(coords_text):
    coords_list = coords_text.split(",")
    coords = []
    for i in range(len(coords_list)):
        coords.append(coords_list[i])
    return np.array(coords).astype("float32")


class GanHands(data.Dataset):
    def __init__(self, directory, num_objects, transform):

        self.directory = directory
        self.labels = [""] * num_objects
        for i in range(len(self.labels)):
            self.labels[i] = "{:04d}".format(i + 1)
        self.__transform = transform
        self.__length = num_objects

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image_file = self.labels[index] + "_color_composed.png"
        relative_coords_file = self.labels[index] + "_joint_pos.txt"
        image = Image.open(os.path.join(self.directory, image_file))
        image = np.asarray(image)
        image = image / 255

        relative_coords_file = open(os.path.join(self.directory, relative_coords_file)).read()
        coords = process_relative_coords_file(relative_coords_file)
        sample = {"image": image, "coords": coords}

        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample
