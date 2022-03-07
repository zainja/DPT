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


NO_OBJECT_SIZE = 142
OBJECT_SIZE = 185
OBJECTS_IN_FILE = 1024


class GanHands(data.Dataset):
    def __init__(self, directory, transform):
        self.directory = directory
        self.labels = []
        for folder in range(1, NO_OBJECT_SIZE):
            for i in range(1, OBJECTS_IN_FILE + 1):
                self.labels.append("noObject/{:04d}/{:04d}".format(folder, i))
        for folder in range(1, OBJECT_SIZE):
            for i in range(1, OBJECTS_IN_FILE + 1):
                self.labels.append("withObject/{:04d}/{:04d}".format(folder, i))

        print(self.labels[:3])
        self.__transform = transform
        self.__length = len(self.labels)

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
